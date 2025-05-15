import numpy as np

class SearchGameLogic:
    """
    Логика "поиска в массиве" как игра для AlphaZero.
    """
    def __init__(self, array, target):
        self.array = list(array)
        self.target = target
        self.first = 0
        self.last = len(self.array) - 1
        self.cur = None
        self.vars = [0] * 5
        self.done = False
        self.found = False
        self.steps = []
        self.allowed = set()

    def reset(self):
        self.cur = (self.first + self.last) // 2
        self.vars = [0] * 5
        self.done = False
        self.found = False
        self.steps = []
        self._recompute_allowed()
        return self.get_obs()

    def _recompute_allowed(self):
        # если последний был 'set', то только числа (1-100)
        if self.steps and isinstance(self.steps[-1], tuple) and self.steps[-1][0] == 'set':
            self.allowed = set(range(1, 101))
        else:
            # базовый набор команд
            cmds = ['cmp', 'set', 'add', 'sub', 'mul', 'div',
                    'ifless', 'ifless_close', 'ifbigger', 'ifbigger_close', 'mov', 'end']
            self.allowed = set(cmds)

    def step(self, action):
        assert action in self.allowed, f"Недопустимый ход: {action}"
        # сохраняем шаг
        self.steps.append(action)

        # выполнение команды
        if action == 'cmp':
            if self.array[self.cur] == self.target:
                self.done = True
                self.found = True
        elif action == 'end':
            self.done = True
        elif isinstance(action, tuple) and action[0] == 'set':
            # ('set', var_idx, value)
            _, idx, val = action
            self.vars[idx] = val
        elif isinstance(action, tuple) and action[0] == 'add':
            _, idx, val = action
            self.vars[idx] += val
        elif isinstance(action, tuple) and action[0] == 'sub':
            _, idx, val = action
            self.vars[idx] -= val
        elif isinstance(action, tuple) and action[0] == 'mul':
            _, idx, val = action
            self.vars[idx] *= val
        elif isinstance(action, tuple) and action[0] == 'div':
            _, idx, val = action
            self.vars[idx] = int(self.vars[idx] / val) if val != 0 else 0
        elif isinstance(action, tuple) and action[0] == 'ifless':
            _, var_idx, val = action
            if self.cur < val:
                pass  # тело if управляется MCTS
        elif action == 'ifless_close':
            pass
        elif isinstance(action, tuple) and action[0] == 'ifbigger':
            _, var_idx, val = action
            if self.cur > val:
                pass
        elif action == 'ifbigger_close':
            pass
        elif isinstance(action, tuple) and action[0] == 'mov':
            _, val = action
            self.cur = val
        # иначе, команды типа чисел 1–100 не используются здесь напрямую

        obs = self.get_obs()
        reward = self.compute_reward() if self.done else 0
        self._recompute_allowed()
        return obs, reward, self.done

    def get_obs(self):
        # вектор: [cur, array[cur], target] + vars + one-hot mask(allowed)
        one_hot = np.zeros(12 + 5 + 100, dtype=np.int8)
        one_hot[0] = self.cur
        one_hot[1] = self.array[self.cur]
        one_hot[2] = self.target
        for i, v in enumerate(self.vars): one_hot[3 + i] = v
        # маска команд
        for cmd in self.allowed:
            if isinstance(cmd, str):
                # хэшим строку в индекс 8..? (упрощённо)
                pass
        return one_hot

    def compute_reward(self):
        # 0: не найдено, +1: найдено, +bonus за эффективность
        if not self.found:
            return -1
        # базовый бонус за нахождение
        bonus = 1
        # оценка сравнений: сравнить len(self.steps_of_cmp)
        cmp_count = sum(1 for s in self.steps if s == 'cmp')
        # baseline: линейный (n) или бинарный (log2 n)
        n = len(self.array)
        linear = n
        binary = int(np.log2(n)) if self.array == sorted(self.array) else linear
        if cmp_count <= binary:
            bonus += 1
        return bonus