from dataclasses import dataclass
from dataclasses import field
from typing import List
from typing import Dict
import math
import random


@dataclass
class Building:
    price: float
    power: float


@dataclass
class Upgrade:
    price: float
    power: float


@dataclass
class State:
    buildings_owned: Dict[str, int] = field(default_factory=dict)
    upgrades_owned: Dict[str, bool] = field(default_factory=dict)
    time: float = 0
    money: float = 0
    highest_money: float = 0
    reward: float = 0


class IdleGame:
    buildings = {
        "generator1": Building(price=150, power=1),
        "generator2": Building(price=1000, power=10),
        "generator3": Building(price=11000, power=80),
        "generator4": Building(price=120000, power=470),
        "generator5": Building(price=1300000, power=2600),
        "generator6": Building(price=1.4e7, power=14000),
        "generator7": Building(price=20e7, power=78000),
        "generator8": Building(price=330e7, power=440000),
        "generator9": Building(price=5.1e10, power=2600000),
        "generator10": Building(price=75e10, power=1.6e7),
        "generator11": Building(price=1e13, power=10e7),
    }

    upgrades = {
        "generator1-1": Upgrade(price=1000, power=2),
        "generator1-2": Upgrade(price=5000, power=2),
        "generator1-3": Upgrade(price=100000, power=2),
        "generator2-1": Upgrade(price=10000, power=2),
        "generator2-2": Upgrade(price=50000, power=2),
        "generator2-3": Upgrade(price=500000, power=2),
        "generator2-4": Upgrade(price=5e7, power=2),
        "generator2-5": Upgrade(price=500e7, power=2),
        "generator2-6": Upgrade(price=50e10, power=2),
        "generator3-1": Upgrade(price=110000, power=2),
        "generator3-2": Upgrade(price=550000, power=2),
        "generator3-3": Upgrade(price=5500000, power=2),
        "generator3-4": Upgrade(price=55e7, power=2),
        "generator3-5": Upgrade(price=5.5e10, power=2),
        "generator3-6": Upgrade(price=550e10, power=2),
        "generator4-1": Upgrade(price=1200000, power=2),
        "generator4-2": Upgrade(price=6000000, power=2),
        "generator4-3": Upgrade(price=6e7, power=2),
        "generator4-4": Upgrade(price=600e7, power=2),
        "generator4-5": Upgrade(price=60e10, power=2),
        "generator5-1": Upgrade(price=1.3e7, power=2),
        "generator5-2": Upgrade(price=6.5e7, power=2),
        "generator5-3": Upgrade(price=65e7, power=2),
        "generator5-4": Upgrade(price=6.5e10, power=2),
        "generator5-5": Upgrade(price=650e10, power=2),
        "generator6-1": Upgrade(price=14e7, power=2),
        "generator6-2": Upgrade(price=70e7, power=2),
        "generator6-3": Upgrade(price=700e7, power=2),
        "generator6-4": Upgrade(price=70e10, power=2),
        "generator7-1": Upgrade(price=200e7, power=2),
        "generator7-2": Upgrade(price=1e10, power=2),
        "generator7-3": Upgrade(price=10e10, power=2),
        "generator7-4": Upgrade(price=3.3e10, power=2),
        "generator7-5": Upgrade(price=16.5e10, power=2),
        "generator7-6": Upgrade(price=165e10, power=2),
        "generator8-1": Upgrade(price=51e10, power=2),
        "generator8-2": Upgrade(price=255e10, power=2),
        "generator8-3": Upgrade(price=750e10, power=2),
    }

    goal: float = 1e11
    click_money: int = 5
    building_buy_buffer: float = 1
    mistakes: int = 0
    state: State

    def __init__(self):
        self.state: State = State()

        for building in self.buildings:
            self.state.buildings_owned[building] = 0

        for upgrade in self.upgrades:
            self.state.upgrades_owned[upgrade] = False

        self.actions = (
            ["nothing"] + list(self.buildings.keys()) + list(self.upgrades.keys())
        )

        self.action_indexes = {}

        for i in range(len(self.actions)):
            self.action_indexes[self.actions[i]] = i

    def total_price(self, base_price: float, level: float):
        return math.floor(base_price * (1.15 ** level))

    def buy_building(self, building: str):
        price = self.total_price(
            self.buildings[building].price, self.state.buildings_owned[building]
        )
        if price * self.building_buy_buffer <= self.state.money:
            self.state.money -= price
            self.state.buildings_owned[building] += 1
            return True
        return False

    def buy_cheapest_building(self):
        for building in self.buildings:
            if self.buy_building(building):
                self.print_state(building)
                return

    def buy_best_ratio_building(self):
        best_building = None
        best_ratio = 0
        for building in self.buildings:
            ratio = self.buildings[building].power / self.total_price(
                self.buildings[building].price, self.state.buildings_owned[building]
            )
            if ratio > best_ratio:
                best_ratio = ratio
                best_building = building
        if self.buy_building(best_building):
            self.print_state(best_building)

    def buy_upgrade(self, upgrade: str):
        if (
            not self.state.upgrades_owned[upgrade]
            and self.upgrades[upgrade].price <= self.state.money
        ):
            self.state.upgrades_owned[upgrade] = True
            self.state.money -= self.upgrades[upgrade].price
            # self.print_state(upgrade)
            return True
        return False

    def buy_cheapest_upgrade(self):
        for upgrade in self.upgrades:
            if self.buy_upgrade(upgrade):
                return

    def calculate_production(self, steps: int):
        for building in self.buildings:
            building_power: float = self.buildings[building].power
            for upgrade in self.state.upgrades_owned:
                if self.state.upgrades_owned[upgrade] and upgrade.startswith(
                    f"{building}-"
                ):
                    building_power *= self.upgrades[upgrade].power
            self.state.money += self.state.buildings_owned[building] * building_power
        self.state.money += self.click_money * steps
        self.state.highest_money = max(self.state.highest_money, self.state.money)

    def get_state(self):
        vals: List[float] = []
        for building in self.state.buildings_owned:
            vals.append(self.state.buildings_owned[building])
        for upgrade in self.state.upgrades_owned:
            vals.append(float(self.state.upgrades_owned[upgrade]))

        vals.append(self.state.money)
        vals.append(self.state.highest_money)
        # vals.append(self.state.time)
        return vals

    def print_state(self, action: str):
        print(self.action_indexes[action] or 0)
        print(self.get_state())

    def get_reward(self):
        return self.state.reward

    def execute_action(self, index: int):
        action = self.actions[index]
        success = True
        if action in self.buildings:
            success = self.buy_building(action)
        elif action in self.upgrades:
            success = self.buy_upgrade(action)
        if not success:
            self.mistakes += 1
            self.state.reward = -1
        else:
            self.state.reward = self.state.highest_money

    def one_hot_actions(self, index: int):
        # print(len(self.actions), index)
        one_hot = [0] * len(self.actions)
        one_hot[index] = 1
        return one_hot

    def valid_actions(self):
        valid_actions = [0]
        for action, index in self.action_indexes.items():
            if action in self.buildings:
                price = self.total_price(
                    self.buildings[action].price, self.state.buildings_owned[action]
                )
                if price <= self.state.money:
                    valid_actions += [index]
            if (
                action in self.upgrades
                and not self.state.upgrades_owned[action]
                and self.upgrades[action].price <= self.state.money
            ):
                valid_actions += [index]
        return valid_actions

    def random_action(self):
        valid_actions = self.valid_actions()
        return valid_actions[random.randint(0, len(valid_actions) - 1)]

    def step(self, steps: int):
        self.calculate_production(steps)
        self.state.time += steps

    def is_over(self):
        return self.state.money < self.goal

    def run(self):
        while self.state.time < 100000:
            # if self.state.time % 1000 == 0:
            #     self.print_state("nothing")

            # self.buy_cheapest_building()
            self.buy_best_ratio_building()
            self.buy_cheapest_upgrade()

            self.step(1)

        self.print_state("nothing")


if __name__ == "__main__":
    game = IdleGame()
    game.run()
