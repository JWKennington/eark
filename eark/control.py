"""Control utiliies
"""
import typing

from eark.state import State


class ControlRule:
    def __init__(self, default: float = 0.0):
        self.default = default

    def __add__(self, other):
        if not isinstance(other, ControlRule):
            raise ValueError('Addition not defined for ControlRule and type: {}'.format(type(other)))
        rules = self.rules if isinstance(self, CompositeControlRule) else [self]
        if isinstance(other, CompositeControlRule):
            rules.extend(other.rules)
        else:
            rules.append(other)
        return CompositeControlRule(rules=rules)

    def rule_applies(self, t: float, state: State):
        raise NotImplementedError

    def drum_omega(self, t: float, state: State):
        raise NotImplementedError


class CompositeControlRule(ControlRule):
    def __init__(self, rules: typing.Tuple[ControlRule]):
        self.rules = rules

    def rule_applies(self, t: float, state: State):
        return any(rule.rule_applies(t, state) for rule in self.rules)

    def drum_omega(self, t: float, state: State):
        return sum(rule.drum_omega(t, state) for rule in self.rules)


class LinearControlRule(ControlRule):
    def __init__(self, coeff: float, const: float, t_min: float = None, t_max: float = None, default: float = 0.0):
        super().__init__(default=default)
        self.coeff = coeff
        self.const = const
        self.t_min = t_min
        self.t_max = t_max

    def __repr__(self):
        t_boundary_string = ('None' if self.t_min is None else '{:.1f}'.format(self.t_min) + ', ') + \
                            ('None' if self.t_max is None else '{:.1f}'.format(self.t_max))
        return 'LinearControlRule({:.1f}, {:.1f}, {})'.format(self.coeff, self.const, t_boundary_string)

    def rule_applies(self, t: float, state: State):
        return ((self.t_min is not None and t >= self.t_min) or self.t_min is None) and \
               ((self.t_max is not None and t <= self.t_max) or self.t_max is None)

    def drum_omega(self, t: float, state: State):
        if self.rule_applies(t, state):
            return self.coeff * t + self.const
        return self.default



