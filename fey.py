# Copyright 2024 Leo Peckham

from __future__ import annotations
from dataclasses import dataclass, field as dataclass_field
from typing import (
        Self, overload, Callable, override,
        Iterator, NewType
        )

def component(value: object) -> Component:
    match value:
        case Component(): return value
        case _: return Lit(value)

def type_repr(t: type | tuple[()] | Fn):
    match t:
        case tuple(): return "tuple[()]"
        case Fn(): return "Fn"
        case _ if hasattr(t, '__name__'): return t.__name__
        case _: raise NotImplementedError


def value_repr(v: object, fn_prefix: str) -> str:
    match v:
        case str():
            return '"' + v.encode("unicode_escape").decode("ASCII") + '"'
        case Fn():
            return SCOPE.add(v, fn_prefix)
        case _ if hasattr(v, '__repr__'): return repr(v)
        case _: raise NotImplementedError

@dataclass
class Scope:
    scope: dict[int, tuple[Fn, str]] = dataclass_field(default_factory=dict)

    def get(self, other: Fn) -> str | None:
        return self.scope.get(id(other), (None, None))[1]

    def add(self, other: Fn, fn_prefix: str) -> str:
        id_ = id(other)
        if id_ in self.scope:
            return self.scope[id_][1]
        else:
            ident = f"{fn_prefix}{len(self.scope)}"
            self.scope[id_] = (other, ident)
            return ident

    def dict(self) -> dict[str, Fn]:
        return {s: f for f, s in self.scope.values()}

SCOPE = Scope()

class Component:
    def __add__(self, other: object) -> Expr:
        return Expr(self, Op('+'), component(other))

    def __radd__(self, other: object) -> Expr:
        return Expr(component(other), Op('+'), self)

    def __sub__(self, other: object) -> Expr:
        return Expr(self, Op('-'), component(other))

    def __mul__(self, other: object) -> Expr:
        return Expr(self, Op('*'), component(other))

    def __rmul__(self, other: object) -> Expr:
        return Expr(component(other), Op('*'), self)

    def __getitem__(self, key: object) -> Expr:
        return Expr(self, GetItem(component(key)))

    def __call__(self, *args: object) -> Expr:
        return Expr(self, Call(*[component(arg) for arg in args]))

@dataclass
class Lit[T](Component):
    value: T
    
    def __repr__(self) -> str:
        return f"lit({self.value})"

@dataclass
class Op(Component):
    value: str

class Call(Component):
    args: list[Component]

    def __init__(self, *args: Component) -> None:
        self.args = list(args)

@dataclass
class GetItem[T](Component):
    key: T

@dataclass
class Arg(Component):
    index: int

    def __repr__(self) -> str:
        return f"arg_{self.index}"

class Expr(Component):
    components: list[Component]

    def __init__(self, *args: Component) -> None:
        self.components = list(args)

    def as_str(self, name: str, arg_prefix: str, fn_prefix: str) -> str:
        def get_str(component: Component) -> str:
            match component:
                case Lit(): return value_repr(component.value, fn_prefix)
                case Op(): return " " + component.value + " "
                case Arg(): return f"{arg_prefix}{component.index}"
                case Call():
                    args = ", ".join([get_str(arg) for arg in component.args])
                    return f"({args})"
                case GetItem():
                    return f"[{get_str(component.key)}]"
                case Expr():
                    return f"({component.as_str(name, arg_prefix, fn_prefix)})"
            assert False, f"Unknown Component {component}"

        components = [get_str(component) for component in self.components]
        return ''.join(components)

    def __repr__(self) -> str:
        return f"Expr( {self.as_str('rec', 'z', 'f')} )"

class Pattern:
    pattern: list[object]
    
    def __init__(self, *args: object) -> None:
        self.pattern = list(args)

    def __repr__(self) -> str:
        return f"Pattern({', '.join(map(repr, self.pattern))})"

# TODO: Fn should type check a lot more strongly
@dataclass
class Fn[*In, Out]:
    cases: dict[Pattern, Expr] = dataclass_field(default_factory=dict)
    types: list[type | tuple[()] | Fn] = dataclass_field(default_factory=list)
    code: str = ""
    
    @staticmethod
    def default_fn(*_: *In) -> Out:
        raise NotImplementedError  # intended behaviour

    fn: Callable[[*In], Out] = default_fn

    def generate_header(self, name: str, prefix: str) -> str:
        *arg_types, ret_type = self.types
        args = [f"{prefix}{i}: {type_repr(t)}" for i, t in enumerate(arg_types)]
        header = f"def {name}({', '.join(args)}) -> {type_repr(ret_type)}:"
        return header

    def generate_pattern(self, pattern: Pattern, arg_prefix: str,
                         fn_prefix: str) -> str:
        items = [f"{arg_prefix}{i}"
                 if isinstance(item, Arg)
                 else value_repr(item, fn_prefix)
                 for i, item in enumerate(pattern.pattern)]
        return ", ".join(items)

    def generate_case(self, pattern: Pattern, expr: Expr,
                      name: str, arg_prefix: str, fn_prefix) -> str:
        pattern_str = self.generate_pattern(pattern, arg_prefix, fn_prefix)
        expr_str = expr.as_str(name, arg_prefix, fn_prefix)
        return f"case {pattern_str}: return {expr_str}"

    def generate_body(self, name: str, arg_prefix: str, fn_prefix: str
                      ) -> list[str]:
        *args, _ = [f"{arg_prefix}{i}" for i in range(len(self.types))]
        match_head = f"match {', '.join(args)}:"
        body = [f"\t{self.generate_case(pattern, expr, name,
                                        arg_prefix, fn_prefix)}"
                for pattern, expr in self.cases.items()]
        return [match_head] + body

    def generate_fn(self, name: str, arg_prefix: str, fn_prefix: str) -> str:
        glob = f"global {name}\n"
        header = self.generate_header(name, arg_prefix) + '\n'
        body = ''.join(f"\t{line}\n"
                       for line in self.generate_body(name, arg_prefix,
                                                      fn_prefix))
        return glob + header + body

    def make_fn(self) -> None:
        name = "fn"
        arg_prefix = "z"
        fn_prefix = "f"

        self.code = self.generate_fn(name, arg_prefix, fn_prefix)

        d = SCOPE.dict()

        exec(self.code, d)

        # we have to type ignore here, because of the way scope works and
        # how the code is generated
        # TODO: make a manual check that fn's annotations fit *In and Out
        self.fn = d[name]  # type: ignore

    # TODO: Currying
    def __call__(self, *args: *In) -> Out | Component:
        if any(isinstance(arg, Component) for arg in args):
            return Lit(self)(*args)
        else:
            return self.fn(*args)

    def __setitem__(self, key: object, value: object) -> None:
        pattern: Pattern
        match key:
            case tuple(): pattern = Pattern(*key)
            case _: pattern = Pattern(key)

        expr: Expr
        match value:
            case Component(): expr = Expr(value)
            case _: expr = Expr(Lit(value))

        self.cases[pattern] = expr
        self.make_fn()

    def __repr__(self) -> str:
        return " -> ".join(map(type_repr, self.types))

@dataclass
class ConstFn(Fn):
    @override
    def generate_header(self, name: str, prefix: str = "") -> str:
        _, ret_type = self.types
        header = f"def {name}() -> {type_repr(ret_type)}:"
        return header

    def calculate_return(self, name: str, expr: Expr) -> str:
        result = eval(expr.as_str(name, "", ""))
        return (result)

    def generate_return(self, name: str, expr: Expr, fn_prefix: str) -> str:
        return f"return {value_repr(self.calculate_return(name, expr), fn_prefix)}"

    @override
    def generate_body(self, name: str, arg_prefix: str, fn_prefix: str
                      ) -> list[str]:
        expr = list(self.cases.values())[-1]
        body = [f"{self.generate_return(name, expr, fn_prefix)}"]
        return body

    @override
    def generate_fn(self, name: str, arg_prefix: str, fn_prefix) -> str:
        header = self.generate_header(name, arg_prefix) + '\n'
        body = ''.join(f"\t{line}\n"
                       for line in self.generate_body(name, arg_prefix,
                                                      fn_prefix))
        return header + body

class ConstFnFactory:
    def __truediv__(self, t: type) -> ConstFn:
        return ConstFn(types=[(), t])

@dataclass
class FnFactory(tuple[Fn, *tuple[Arg, ...]]):
    types: list[type | tuple[()] | Fn] = dataclass_field(default_factory=list)
    
    @overload
    def __floordiv__(self, t: type | FnFactory) -> Self:
        ...
    @overload
    def __floordiv__(self, t: tuple[()]) -> ConstFnFactory:
        ...

    # implementation
    def __floordiv__(self, t):
        self.types = []
        match t:
            case FnFactory():
                self.types.append(Fn(types=t.types))
                return self
            case type():
                self.types.append(t)
                return self
            case ():
                return ConstFnFactory()
        assert False, f"Unknown type {t}"

    def __truediv__(self, t: type) -> Self:
        self.types.append(t)
        return self

    @override
    def __iter__(self) -> Iterator[Fn | Arg]:
        return iter([Fn(types=self.types)] +
                    [Arg(i) for i in range(len(self.types) - 1)])

fn = FnFactory()

if __name__ == "__main__":
    hello = fn // () / str
    hello [()] = "Hello, world!"

    assert hello() == "Hello, world!"

    add, x, y = fn // int / int / int
    add [x, y] = x + y

    assert add(3, 5) == 8

    greet, name = fn // str / str
    greet ["Bob"] = "Long time no see, ol' pal!"
    greet [name] = "Nice to meet you, " + name

    assert greet("John") == "Nice to meet you, John"

    inc, n = fn // int / int
    inc [n] = n + 1
    
    assert inc(0) == 1

    f, x, y = fn // int / int / int
    f [x, y] = add(2 * x, 2 * y) + 1

    assert f(3, 4) == 15

    fib, n = fn // int / int
    fib [0] = 0
    fib [1] = 1
    fib [n] = fib(n - 1) + fib(n - 2)

    assert fib(6) == 8

    alpha, a = fn // int / int
    beta, b = fn // int / int

    alpha [0] = 0
    alpha [a] = beta(a - 1) + 1

    beta [0] = 0
    beta [b] = 2 * alpha(b - 1)

    # beta(5) = 2 * alpha(4)
    #         = 2 * (beta(3) + 1)
    #         = 2 * (2 * alpha(2) + 1)
    #         = 2 * (2 * (beta(1) + 1) + 1)
    #         = 2 * (2 * (2 * alpha(0) + 1) + 1)
    #         = 2 * (2 * (2 * 0 + 1) + 1)
    #         = 6
    assert beta(5) == 6

    # TODO: Better generics
    a = type(NewType('a', object))
    b = type(NewType('b', object))
    foldr, f, z, xs = fn // (fn // a / b / b) / a / list[b] / b
    foldr [f, z, []] = z
    foldr [f, z, xs] = f(xs[0], foldr(f, z, xs[1:]))

    assert foldr(add, 0, [1,2,3,4,5]) == 15

