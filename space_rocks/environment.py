import numpy as np
import pygame
from game import SpaceRocks

class AsteroidEnv:
    def __init__(self, render_mode=False):
        self.render_mode = render_mode
        self.game = SpaceRocks()
        self.action_space = 5  # [0: do nothing, 1: rotate left, 2: rotate right, 3: accelerate, 4: shoot]
        self.observation_space = 10  # You can adjust based on features

    def reset(self):
        self.game = SpaceRocks()
        obs = self.get_state()
        return obs

    def step(self, action):
        self._apply_action(action)
        self.game._process_game_logic()
        obs = self.get_state()
        reward, done = self._get_reward_and_done()
        return obs, reward, done, {}

    def render(self):
        if self.render_mode:
            self.game._draw()

    def get_state(self):
        ship = self.game.spaceship
        if not ship:
            return np.zeros(self.observation_space, dtype=np.float32)

        asteroid_data = []
        for asteroid in self.game.asteroids[:3]:  # max 3 asteroids
            asteroid_data += [*asteroid.position, *asteroid.velocity]

        while len(asteroid_data) < 6:
            asteroid_data += [0, 0]  # padding if fewer than 3

        ship_data = [
            *ship.position,
            *ship.velocity,
            *ship.direction,
        ]

        return np.array(ship_data + asteroid_data[:6], dtype=np.float32)

    def _apply_action(self, action):
        ship = self.game.spaceship
        if not ship:
            return

        if action == 1:
            ship.rotate(clockwise=False)
        elif action == 2:
            ship.rotate(clockwise=True)
        elif action == 3:
            ship.accelerate()
        elif action == 4:
            ship.shoot()

    def _get_reward_and_done(self):
        if not self.game.spaceship:
            return -10, True  # Lost
        elif not self.game.asteroids:
            return 20, True  # Win
        else:
            return 0.1, False  # Encourage survival
