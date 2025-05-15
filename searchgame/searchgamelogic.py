import numpy as np

class SearchGameLogic:
    # Объявляем список всех строковых команд один раз в классе
    cmds = [
        'cmp', 'set', 'add', 'sub', 'mul', 'div',
        'ifless', 'ifless_close', 'ifbigger', 'ifbigger_close',
        'mov', 'end'
    ]

    def __init__(self, array, target, cur=None, vars=None, allowed=None, done=False, found=False, cmp_count=0):
        self.array = list(array)
        self.target = target

        # Инициализация указателя
        self.first = 0
        self.last = len(self.array) - 1
        self.cur = cur if cur is not None else (self.first + self.last) // 2

        # Пять вспомогательных переменных
        self.vars = vars.copy() if vars is not None else [0] * 5

        # История шагов
        self.steps = []

        # Флаги состояния
        self.done = done
        self.found = found
        self.cmp_count = cmp_count  # сколько раз вызывался `cmp`

        # Маска допустимых ходов
        self.allowed = allowed if allowed is not None else set(range(1, 101)) | set(self.cmds)

    @classmethod
    def get_action_size(cls):
        return 100 + len(cls.cmds)

    def reset(self):
        """Принудительно переинициализировать игру."""
        self.cur = (self.first + self.last) // 2
        self.vars = [0] * 5
        self.steps = []
        self.done = False
        self.found = False
        self.cmp_count = 0
        self.recompute_allowed()

    def recompute_allowed(self):
        """Пересчитывает self.allowed на основании последнего хода."""
        last = self.steps[-1] if self.steps else None
        if last == 'set':
            # после set можно только выбирать числа
            self.allowed = set(range(1, 101))
        else:
            # в остальных случаях все команды
            self.allowed = set(range(1, 101)) | set(self.cmds)

    def step(self, action):
        """Делает один ход: action может быть int 1–100 или строковая команда."""
        assert not self.done, "Ход после окончания игры"
        assert action in self.allowed, f"Недопустимый ход: {action}"

        # обновляем историю
        self.steps.append(action)

        # логика выполнения
        if action == 'cmp':
            self.cmp_count += 1
            if self.array[self.cur] == self.target:
                self.done = True
                self.found = True
        elif action == 'end':
            self.done = True
        elif isinstance(action, int):
            # простейшая команда: mov cur на число
            self.cur = action - 1
        else:
            # команды с двумя аргументами, например ('add', var_idx, value)
            cmd = action
            # предполагаем, что action уже разобран на нужные части заранее
            # здесь приведён упрощённый пример:
            # если cmd == 'set': self.vars[var_idx] = value
            pass

        # пересчёт доступных ходов
        self.recompute_allowed()

        # возвращаем новое наблюдение, награду и признак окончания
        return self.get_obs(), self.compute_reward(), self.done

    def compute_reward(self):
        """0 вне окончания; +1 за нахождение; + bonus за эффективность."""
        if not self.done:
            return 0.0
        if not self.found:
            return -1.0
        # bonus: если cmp_count <= линейный или бинарный порог
        # допустим, линейный = len(array), бинарный = log2(len)
        linear_threshold = len(self.array)
        binary_threshold = int(np.log2(len(self.array))) + 1
        thresh = linear_threshold if not self.is_sorted() else binary_threshold
        bonus = 1.0 if self.cmp_count <= thresh else 0.0
        return 1.0 + bonus

    def is_sorted(self):
        """Проверяет, отсортирован ли массив (для задачи бинарного поиска)."""
        return all(self.array[i] <= self.array[i+1] for i in range(len(self.array)-1))

    def get_obs(self):
        """
        Векторизуем состояние:
        [array..., target, cur, vars[5], mask_allowed...]
        mask_allowed — длина = 100 + len(cmds).
        """
        obs = []
        # сам массив и таргет
        obs.extend(self.array)
        obs.append(self.target)
        # указатель и 5 переменных
        obs.append(self.cur)
        obs.extend(self.vars)
        # маска доступных ходов
        total_actions = 100 + len(self.cmds)
        mask = [0] * total_actions
        for action in self.allowed:
            if isinstance(action, int) and 1 <= action <= 100:
                mask[action - 1] = 1
            elif action in self.cmds:
                idx = 100 + self.cmds.index(action)
                mask[idx] = 1
        obs.extend(mask)
        return np.array(obs, dtype=np.float32)

    @classmethod
    def from_obs(cls, obs):
        """
        Восстанавливаем логику из вектора obs.
        Формат obs: [array..., target, cur, vars5, mask_allowed...]
        """
        n = len(obs)
        # Длина массива = (n - 1 таргет -1 cur -5 vars - mask)/?
        # Предположим, вы заранее фиксируете длину массива M и mask_len = 100+len(cmds)
        M = cls._array_length
        mask_len = 100 + len(cls.cmds)

        array = obs[:M].astype(int).tolist()
        target = int(obs[M])
        cur = int(obs[M + 1])
        vars = list(obs[M + 2:M + 7].astype(int))
        allowed_mask = obs[-mask_len:].astype(int).tolist()

        # строим множесто allowed
        allowed = set()
        for i, bit in enumerate(allowed_mask):
            if bit:
                if i < 100:
                    allowed.add(i + 1)
                else:
                    allowed.add(cls.cmds[i - 100])

        return cls(array=array, target=target,
                   cur=cur, vars=vars, allowed=allowed,
                   done=False, found=False, cmp_count=0)
