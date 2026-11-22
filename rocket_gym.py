"""六自由度火箭动力着陆环境（PPO 训练用 Gymnasium 环境）。

动力学模型（状态方程 f_func、四元数运动学、质量流模型等）参考并改编自
PythonRobotics 项目的火箭着陆示例：

    https://github.com/AtsushiSakai/PythonRobotics
    License: MIT
    Author: Atsushi Sakai (@Atsushi_twi)

在原动力学基础上的主要改动：
- 封装为 Gymnasium 环境；
- 动作空间改为 4 维：推力大小、喷口倾角、喷口方位角的 cos/sin；
- 增加引导奖励、终端速度惩罚、下滑道约束等训练相关逻辑；
- 整理渲染与数值代码（四元数归一化、seed 复现等）。
"""

from typing import Optional

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from gymnasium import register, spaces


class Rocket_Model_6DoF(gym.Env):
    """六自由度火箭动力着陆环境。

    状态为质量、位置、速度、四元数和机体角速度，共 14 维。
    动作为推力大小、喷口倾角、喷口方位角的 cos/sin，共 4 维。
    """

    def __init__(self, rng: Optional[np.random.Generator] = None):
        self.rng = rng
        self.STEP = 0.1
        self.n_x = 14
        self.n_u = 4

        self.m_wet = 3.0
        self.m_dry = 2.2
        self.t_f_guess = 10.0

        self.r_I_final = np.array([0.0, 0.0, 0.0])
        self.v_I_final = np.array([-0.1, 0.0, 0.0])
        self.w_B_max = np.deg2rad(60.0)

        self.max_gimbal = np.deg2rad(20.0)
        self.cos_theta_max = np.cos(np.deg2rad(90.0))
        self.tan_gamma_gs = np.tan(np.deg2rad(30.0))
        self.glide_slope_min_height = 0.3

        self.T_max = 5.0
        self.T_min = 0.3
        self.J_B = 1e-2 * np.diag([1.0, 1.0, 1.0])
        self.J_B_inv = np.linalg.inv(self.J_B)
        self.g_I = np.array([-1.0, 0.0, 0.0])
        self.alpha_m = 0.01
        self.r_T_B = np.array([-1e-2, 0.0, 0.0])

        self.weight_velocity_heuristic = -0.1
        self.weight_tilt_heuristic = 0.05
        self.weight_fuel = 5.0
        self.weight_position = -2.0
        self.weight_tilt = 1.0
        self.weight_w_B = -1.0
        self.weight_velocity_terminal = -2.0

        low_action = np.array(
            [self.T_min, 0.0, -1.0, -1.0], dtype=np.float32
        )
        high_action = np.array(
            [self.T_max, self.max_gimbal, 1.0, 1.0], dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=low_action, high=high_action, dtype=np.float32
        )
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(self.n_x,), dtype=np.float32
        )

        self.reset()

    def set_random_initial_state(self):
        rng = self.rng if self.rng is not None else self.np_random

        self.r_I_init = np.array([
            rng.uniform(3.0, 4.0),
            *rng.uniform(-2.0, 2.0, size=2),
        ])
        self.v_I_init = np.empty(3)
        self.v_I_init[0] = rng.uniform(-1.0, -0.5)
        self.v_I_init[1:3] = (
            rng.uniform(-0.5, -0.2, size=2) * self.r_I_init[1:3]
        )
        self.q_B_I_init = self.euler_to_quat([
            0.0,
            rng.uniform(-30.0, 30.0),
            rng.uniform(-30.0, 30.0),
        ])
        self.w_B_init = np.deg2rad([
            0.0,
            rng.uniform(-20.0, 20.0),
            rng.uniform(-20.0, 20.0),
        ])

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        self.set_random_initial_state()

        self.x = np.concatenate([
            [self.m_wet],
            self.r_I_init,
            self.v_I_init,
            self.q_B_I_init,
            self.w_B_init,
        ])
        self.v0_init = np.linalg.norm(self.v_I_init)
        self._update_derived_state()

        self.x_render = []
        self.u_render = []
        return self.x.astype(np.float32), {}

    def _update_derived_state(self):
        self.remain_fuel = self.x[0] - self.m_dry
        self.consume_fuel = self.m_wet - self.x[0]
        self.tilt = 1.0 - 2.0 * (self.x[9] ** 2 + self.x[10] ** 2)
        horizontal_distance = np.linalg.norm(self.x[2:4])
        self.d_glide_slope = (
            self.x[1] - horizontal_distance * self.tan_gamma_gs
        )
        self.w_B_magnitude = np.linalg.norm(self.x[11:14])

    def step(self, action):
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        if action.shape != (self.n_u,):
            raise ValueError(
                f"Expected action shape ({self.n_u},), got {action.shape}"
            )

        thrust, gimbal, cos_theta, sin_theta = action
        direction_norm = np.hypot(cos_theta, sin_theta)
        if direction_norm < 1e-8:
            cos_theta, sin_theta = 1.0, 0.0
        else:
            cos_theta /= direction_norm
            sin_theta /= direction_norm

        u = np.array([
            thrust * np.cos(gimbal),
            thrust * np.sin(gimbal) * cos_theta,
            thrust * np.sin(gimbal) * sin_theta,
        ])

        self.x = self.x + self.f_func(self.x, u) * self.STEP
        self.x[7:11] /= np.linalg.norm(self.x[7:11])
        self._update_derived_state()
        self.x_render.append(self.x.tolist())
        self.u_render.append(u.tolist())

        reward = self.calc_reward_heuristic(self.x, self.v0_init)
        termination_reasons = []
        if self.x[1] <= 0.0:
            termination_reasons.append("ground_contact")
        if self.remain_fuel < 0.0:
            termination_reasons.append("out_of_fuel")
        if (
            self.x[1] > self.glide_slope_min_height
            and self.d_glide_slope < 0.0
        ):
            termination_reasons.append("glide_slope_violation")
        if self.tilt < self.cos_theta_max:
            termination_reasons.append("tilt_violation")
        if self.w_B_magnitude > self.w_B_max:
            termination_reasons.append("angular_rate_violation")

        terminated = bool(termination_reasons)
        if terminated:
            velocity_error = np.linalg.norm(self.x[4:7] - self.v_I_final)
            reward += (
                -self.weight_fuel * self.consume_fuel
                + self.weight_position * np.linalg.norm(self.x[1:4])
                + self.weight_tilt * self.tilt
                + self.weight_w_B * self.w_B_magnitude
                + self.weight_velocity_terminal * velocity_error ** 2
            )

        obs = self.x.astype(np.float32)
        info = {"termination_reasons": termination_reasons}
        return obs, float(reward), terminated, False, info

    def f_func(self, x, u):
        m = x[0]
        v = x[4:7]
        q0, q1, q2, q3 = x[7:11]
        wx, wy, wz = x[11:14]
        w = x[11:14]

        C_B_I = self.quat_to_rot_matrix(x[7:11])
        acceleration = self.g_I + C_B_I.T @ u / m
        q_dot = 0.5 * np.array([
            -q1 * wx - q2 * wy - q3 * wz,
            q0 * wx + q2 * wz - q3 * wy,
            q0 * wy - q1 * wz + q3 * wx,
            q0 * wz + q1 * wy - q2 * wx,
        ])
        torque = np.cross(self.r_T_B, u)
        w_dot = self.J_B_inv @ (torque - np.cross(w, self.J_B @ w))

        return np.concatenate([
            [-self.alpha_m * np.linalg.norm(u)],
            v,
            acceleration,
            q_dot,
            w_dot,
        ])

    @staticmethod
    def quat_to_rot_matrix(q):
        qw, qx, qy, qz = q
        return np.array([
            [
                1.0 - 2.0 * (qy ** 2 + qz ** 2),
                2.0 * (qx * qy + qw * qz),
                2.0 * (qx * qz - qw * qy),
            ],
            [
                2.0 * (qx * qy - qw * qz),
                1.0 - 2.0 * (qx ** 2 + qz ** 2),
                2.0 * (qy * qz + qw * qx),
            ],
            [
                2.0 * (qx * qz + qw * qy),
                2.0 * (qy * qz - qw * qx),
                1.0 - 2.0 * (qx ** 2 + qy ** 2),
            ],
        ])

    @staticmethod
    def euler_to_quat(angles):
        roll, pitch, yaw = np.deg2rad(angles)
        cr, sr = np.cos(roll * 0.5), np.sin(roll * 0.5)
        cp, sp = np.cos(pitch * 0.5), np.sin(pitch * 0.5)
        cy, sy = np.cos(yaw * 0.5), np.sin(yaw * 0.5)

        return np.array([
            cy * cr * cp + sy * sr * sp,
            cy * sr * cp - sy * cr * sp,
            sy * cr * cp - cy * sr * sp,
            cy * cr * sp + sy * sr * cp,
        ])

    def calc_reward_heuristic(self, obs, v0):
        r = obs[1:4]
        v = obs[4:7]
        tilt = 1.0 - 2.0 * (obs[9] ** 2 + obs[10] ** 2)

        gate_height = 0.5
        gate_weight = np.clip(r[0] / gate_height, 0.0, 1.0)
        gate_position = np.array([gate_height, 0.0, 0.0])
        gate_velocity = np.array([-0.2, 0.0, 0.0])
        low_position = np.array([self.r_I_final[0], r[1], r[2]])

        rt = gate_weight * gate_position + (1.0 - gate_weight) * low_position
        vt = gate_weight * gate_velocity + (1.0 - gate_weight) * self.v_I_final
        tau = gate_weight * 0.2 + (1.0 - gate_weight)

        position_error = r - rt
        distance = np.linalg.norm(position_error)
        if distance < 1e-8:
            v_target = vt.copy()
        else:
            closing_speed = -np.dot(position_error, v) / distance
            effective_speed = max(closing_speed, np.linalg.norm(vt), 1e-3)
            tgo = min(distance / effective_speed, self.t_f_guess)
            velocity_blend = np.exp(-tgo / tau)
            v_target = (
                -v0 * position_error * (1.0 - velocity_blend) / distance
                + velocity_blend * vt
            )

        velocity_reward = (
            self.weight_velocity_heuristic * np.linalg.norm(v - v_target)
        )
        tilt_reward = -self.weight_tilt_heuristic * (1.0 - tilt)
        return velocity_reward + tilt_reward

    @staticmethod
    def axis3d_equal(X, Y, Z, ax):
        max_range = np.array([
            X.max() - X.min(),
            Y.max() - Y.min(),
            Z.max() - Z.min(),
        ]).max()
        grid = np.mgrid[-1:2:2, -1:2:2, -1:2:2]
        Xb = 0.5 * max_range * grid[0].flatten() + 0.5 * (X.max() + X.min())
        Yb = 0.5 * max_range * grid[1].flatten() + 0.5 * (Y.max() + Y.min())
        Zb = 0.5 * max_range * grid[2].flatten() + 0.5 * (Z.max() + Z.min())
        for xb, yb, zb in zip(Xb, Yb, Zb):
            ax.plot([xb], [yb], [zb], "w")

    def render(self, pause: float = 0.5, block: bool = True):
        if not self.x_render:
            return

        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        fig.canvas.mpl_connect(
            "key_release_event",
            lambda event: plt.close(fig) if event.key == "escape" else None,
        )

        X = np.array(self.x_render).T
        U = np.array(self.u_render).T
        print(f"执行了{X.shape[1]}步")

        for k in range(X.shape[1]):
            if not plt.fignum_exists(fig.number):
                break

            plt.cla()
            ax.plot(X[2, :], X[3, :], X[1, :])
            ax.scatter([0.0], [0.0], [0.0], c="r", marker="x")
            self.axis3d_equal(X[2, :], X[3, :], X[1, :], ax)

            rx, ry, rz = X[1:4, k]
            C_B_I = self.quat_to_rot_matrix(X[7:11, k])
            Fx, Fy, Fz = C_B_I.T @ U[:, k]
            dx, dy, dz = C_B_I.T @ np.array([1.0, 0.0, 0.0])

            ax.quiver(
                ry, rz, rx, dy, dz, dx,
                length=0.5,
                linewidth=3.0,
                arrow_length_ratio=0.0,
                color="black",
            )
            ax.quiver(
                ry, rz, rx, -Fy, -Fz, -Fx,
                length=0.1,
                arrow_length_ratio=0.0,
                color="red",
            )
            ax.set_title(f"Rocket powered landing (step {k + 1}/{X.shape[1]})")
            plt.pause(pause)

        if block and plt.fignum_exists(fig.number):
            plt.show()


def main():
    register(
        id="rocket_environment-v0",
        entry_point="rocket_gym:Rocket_Model_6DoF",
    )
    env = gym.make("rocket_environment-v0")

    try:
        for episode in range(10):
            _, _ = env.reset()
            terminated = truncated = False
            episode_return = 0.0

            while not (terminated or truncated):
                action = env.action_space.sample()
                _, reward, terminated, truncated, _ = env.step(action)
                episode_return += reward

            print(f"Episode {episode + 1}: return={episode_return:.3f}")
            env.render()
    finally:
        env.close()


if __name__ == "__main__":
    main()
