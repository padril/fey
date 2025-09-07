# Copyright 2025 Leo Peckham

from __future__ import annotations
from dataclasses import dataclass, field as dataclass_field
from typing import Callable, override, Iterator, get_args, overload
from types import (
        new_class, GenericAlias, BuiltinFunctionType, LambdaType, FunctionType
        )
from abc import ABC, abstractmethod
import sys
import inspect
import builtins
import typeguard
import tempfile
import importlib.util
from functools import partial

def component(value: object) -> Component:
    match value:
        case Component(): return value
        case _: return Lit(value)

def type_repr(t: type | GenericAlias | tuple[()] | Fn):
    match t:
        case GenericAlias():
            args = ', '.join(type_repr(a) for a in get_args(t))
            return f"{t.__name__}[{', '.join(args)}]"
        case tuple(): return "tuple[()]"
        case Fn(): return "Fn"
        case _ if hasattr(t, '__name__'): return t.__name__
        case _: raise NotImplementedError


def value_repr(v: object, fn_prefix: str) -> str:
    match v:
        case slice():
            s = f"{v.start or ''}:{v.stop or ''}"
            if v.step: s += f":{v.step}"
            return s
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

class Component(ABC):
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
        if isinstance(key, slice):
            return Expr(self, Slice(component(key.start),
                                    component(key.stop),
                                    component(key.step)))
        else:
            return Expr(self, GetItem(component(key)))

    def __call__(self, *args: object) -> Expr:
        return Expr(self, Call(*[component(arg) for arg in args]))

    @abstractmethod
    def replace(self, index: int, value: Component) -> Component:
        ...

    @abstractmethod
    def downindex(self, by: int) -> Component:
        ...

@dataclass
class Lit[T](Component):
    value: T
    
    def __repr__(self) -> str:
        return f"lit({self.value})"

    def replace(self, index: int, value: Component) -> Lit:
        return self

    def downindex(self, by: int) -> Lit:
        return self

@dataclass
class Op(Component):
    value: str

    def replace(self, index: int, value: Component) -> Op:
        return self

    def downindex(self, by: int) -> Op:
        return self

class Call(Component):
    args: list[Component]

    def __init__(self, *args: Component) -> None:
        self.args = list(args)

    def replace(self, index: int, value: Component) -> Call:
        return Call(*[a.replace(index, value) for a in self.args])

    def downindex(self, by: int) -> Call:
        return Call(*[a.downindex(by) for a in self.args])

@dataclass
class GetItem[T](Component):
    key: T

    def replace(self, index: int, value: Component) -> GetItem:
        if isinstance(self.key, Component):
            return GetItem(self.key.replace(index, value))
        else:
            return self

    def downindex(self, by: int) -> GetItem:
        if isinstance(self.key, Component):
            return GetItem(self.key.downindex(by))
        else:
            return self

@dataclass
class Slice[T, U, V](Component):
    start: T
    stop: U
    step: V

    def replace(self, index: int, value: Component) -> Slice:
        start = self.start
        if isinstance(start, Component):
            start = start.replace(index, value)
        stop = self.stop
        if isinstance(stop, Component):
            stop = stop.replace(index, value)
        step = self.step
        if isinstance(step, Component):
            step = step.replace(index, value)
        return Slice(start, stop, step)

    def downindex(self, by: int) -> Slice:
        start = self.start
        if isinstance(start, Component):
            start = start.downindex(by)
        stop = self.stop
        if isinstance(stop, Component):
            stop = stop.downindex(by)

        step = self.step
        if isinstance(step, Component):
            step = step.downindex(by)
        return Slice(start, stop, step)

@dataclass
class Arg(Component):
    index: int

    def __repr__(self) -> str:
        return f"arg_{self.index}"

    def replace[T: Component](self, index: int, value: T) -> T | Arg:
        if self.index == index:
            return value
        else:
            return self

    def downindex(self, by: int) -> Arg:
        return Arg(self.index - by)

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
                case Slice():
                    start = get_str(component.start) if component.start else ""
                    stop = get_str(component.stop) if component.stop else ""
                    step = get_str(component.step) if component.step else ""
                    return f"[{start}:{stop}:{step}]"
                case Expr():
                    return f"({component.as_str(name, arg_prefix, fn_prefix)})"
            assert False, f"Unknown Component {component}"

        components = [get_str(component) for component in self.components]
        return ''.join(components)

    def __repr__(self) -> str:
        return f"Expr( {self.as_str('rec', 'z', 'f')} )"

    def replace(self, index: int, value: Component) -> Expr:
        return Expr(*[c.replace(index, value) for c in self.components])

    def downindex(self, by: int) -> Expr:
        return Expr(*[c.downindex(by) for c in self.components])

class Pattern:
    pattern: list[object]
    
    def __init__(self, *args: object) -> None:
        self.pattern = list(args)

    def __repr__(self) -> str:
        return f"Pattern({', '.join(map(repr, self.pattern))})"

@dataclass
class Fn[*In, Out]:
    cases: dict[Pattern, Expr] = dataclass_field(default_factory=dict)
    types: list[type | GenericAlias | tuple[()] | Fn] = dataclass_field(default_factory=list)
    generics: set[str] = dataclass_field(default_factory=set)
    code: str = ""
    saved: list[object] = dataclass_field(default_factory=list)
    
    @staticmethod
    def default_fn(*_: *In) -> Out:
        raise NotImplementedError  # intended behaviour

    fn: Callable[[*In], Out] = default_fn

    def generate_header(self, name: str, prefix: str) -> str:
        *arg_types, ret_type = self.types
        args = [f"{prefix}{i}: {type_repr(t)}" for i, t in enumerate(arg_types)]
        generics = f"[{', '.join(self.generics)}]" if self.generics else ""
        header = f"def {name}{generics}({', '.join(args)}) -> {type_repr(ret_type)}:"
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
        name = "fnc"
        arg_prefix = "z"
        fn_prefix = "f"
        module_name = "_anon_"

        self.code = self.generate_fn(name, arg_prefix, fn_prefix)
        
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write(self.code)
            fname = f.name

        spec = importlib.util.spec_from_file_location(module_name, fname)
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module

        module.__dict__.update(SCOPE.dict())
        module.__dict__.update({'Fn': Fn})

        spec.loader.exec_module(module)

        fn = getattr(module, name)
        self.fn = typeguard.typechecked(fn)

    @overload
    def __call__(self, *args: Component) -> Expr:
        ...

    @overload
    def __call__(self, *args: *In) -> Out | Fn:
        ...

    def __call__(self, *args):
        if any(isinstance(arg, Component) for arg in args):
            return Expr(Lit(self), Call(*args))
        elif self.fn == self.default_fn:
            self.saved += args
            return self
        elif len(self.types) == 2 and self.types[0] == ():
            # We gotta type ignore this because *In is so finicky
            return self.fn(*args) # type: ignore
        elif len(args) < len(self.types) - 1:
            cp = Fn()
            n = len(args)
            cp.types = self.types[n:]
            cp.generics = self.generics - set(getattr(t, "__name__")
                                              for t in self.types[:n]
                                              if hasattr(t, "__name__"))
            if not self.cases:
                # this was a builtin or other weird thing
                cp.fn = partial(self.fn, *args)
            else:
                for pat, cmp in self.cases.items():
                    new_pat = Pattern(*[
                        val.downindex(n) if isinstance(val, Arg) else val
                        for val in pat.pattern[n:]
                        ])
                    new_cmp = cmp
                    for i in range(n):
                        new_cmp = new_cmp.replace(i, Lit(args[i]))
                    new_cmp = new_cmp.downindex(n)
                    cp.cases[new_pat] = new_cmp
                cp.make_fn()
            return cp
        else:
            # We gotta type ignore this because *In is so finicky
            return self.fn(*args) # type: ignore

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
        return " -> ".join(f"({repr(t)})"
                           if isinstance(t, Fn)
                           else type_repr(t)
                           for t in self.types)

    def __rmatmul__[U](self, other: Callable[..., U]) -> U | Fn[object, U] | Component:
        f = fn._call_fn(other)
        if self.fn == self.default_fn:
            return f(*self.saved)
        else:
            return f(self.fn)

    def __or__(self, other: Fn) -> Out | Fn:
        return other(self)

    # composition
    def __and__(self, other: Fn) -> Fn:
        f = Fn()
        res = self.types[-1]
        assert len(other.types) >= 0
        fst = other.types[0]
        # TODO: need a better way to compare two type signatures
        if isinstance(fst, GenericAlias):
            # replace all instances of fst with res in other
            ...
        elif isinstance(fst, Fn):
            # replace all instances ...
            ...
        elif isinstance(res, fst):
            ...
        f.types = self.types[:-1] + other.types[1:]
        # TODO: this needs to be better
        def cls(*args):
            return self.fn(other.fn(*args))
        f.fn = cls
        return f

    def __ror__(self, *other: *In) -> Out | Fn:
        return self(*other)

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
    def __getattr__(self, t: str) -> ConstFn:
        frame = inspect.currentframe()
        assert frame
        caller_frame = frame.f_back
        assert caller_frame
        builtin_types = {name: obj for name, obj in vars(builtins).items()
                         if isinstance(obj, type)}
        local_types = {name: obj for name, obj in caller_frame.f_locals.items()
                       if isinstance(obj, type)}
        types = builtin_types | local_types

        if t in types:
            return ConstFn(types=[(), types[t]])
        else:
            # support generics
            raise NotImplementedError

@dataclass
class FnFactory(tuple[Fn, *tuple[Arg, ...]]):
    types: list[type | GenericAlias | tuple[()] | Fn] = dataclass_field(default_factory=list)
    generics: set[str] = dataclass_field(default_factory=set)
    make_new: bool = True
    
    @property
    def void(self) -> ConstFnFactory:
        return ConstFnFactory()

    def _(self, t: FnFactory) -> FnFactory:
        if self.make_new:
            factory = FnFactory()
            factory.make_new = False
        else:
            factory = self
        factory.types.append(Fn(types=t.types.copy()))
        return factory

    @property
    def of(self):
        factory = self
        class Cls:
            def __getattr__(self, t: str) -> FnFactory:
                frame = inspect.currentframe()
                assert frame
                caller_frame = frame.f_back
                assert caller_frame
                builtin_types = {name: obj for name, obj in vars(builtins).items()
                                 if isinstance(obj, type)}
                local_types = {name: obj for name, obj in caller_frame.f_locals.items()
                               if isinstance(obj, type)}
                types = builtin_types | local_types

                if t in types:
                    u = factory.types[-1]
                    assert not isinstance(u, Fn) and hasattr(u, "__class_getitem__")
                    factory.types[-1] = u.__class_getitem__(types[t])
                    return factory
                else:
                    u = factory.types[-1]
                    assert not isinstance(u, Fn) and hasattr(u, "__class_getitem__")
                    factory.types[-1] = u.__class_getitem__(new_class(t))
                    factory.generics |= {t}
                    return factory
        return Cls()

    def __getattr__(self, t: str) -> FnFactory:
        factory: FnFactory
        if self.make_new:
            factory = FnFactory()
            factory.make_new = False
        else:
            factory = self
        frame = inspect.currentframe()
        assert frame
        caller_frame = frame.f_back
        assert caller_frame
        builtin_types = {name: obj for name, obj in vars(builtins).items()
                         if isinstance(obj, type)}
        local_types = {name: obj for name, obj in caller_frame.f_locals.items()
                       if isinstance(obj, type)}
        types = builtin_types | local_types

        if t in types:
            factory.types.append(types[t])
            return factory
        else:
            factory.types.append(new_class(t))
            factory.generics |= {t}
            return factory

    def __iter__(self) -> Iterator[Fn | Arg]:
        return iter([Fn(types=self.types.copy(), generics=self.generics.copy())] +
                    [Arg(i) for i in range(len(self.types) - 1)])

    def _call_fn[*In, Out](self, f_in: Callable[[*In], Out]) -> Fn[*In, Out]:
        f = Fn[*In, Out]()
        if (isinstance(f_in, BuiltinFunctionType)
            or isinstance(f_in, LambdaType)):
            f.fn = f_in
        else:
            f.fn = typeguard.typechecked(f_in)
        if not isinstance(f.fn, FunctionType):
            sig = inspect.signature(f.fn.__call__)
        else:
            sig = inspect.signature(f.fn)
        f.cases = {}
        if self.types:
            f.types = self.types.copy()
            return f
        for p in sig.parameters.values():
            if p.kind not in {p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD}:
                # TODO: We need to be able to handle non-positional entries
                continue
            if p.annotation == inspect._empty:
                t = f"t{len(f.generics)}"
                f.generics |= {t}
                f.types.append(new_class(t))
            else:
                f.types.append(p.annotation)
        f.types.append(sig.return_annotation)
        return f

    def _call_save(self, args: tuple[object, ...]) -> Fn:
        f = Fn()
        f.saved += args
        return f

    def __call__(self, *args: object) -> Fn:
        if len(args) == 1 and isinstance(args[0], Callable):
            return self._call_fn(args[0])
        else:
            return self._call_save(args)

    def __rmatmul__(self, other: Callable) -> Fn:
        f = self._call_fn(other)
        return f

fn = FnFactory()

if __name__ == "__main__":
    hello = fn . void . str
    hello [()] = "Hello, world!"

    assert hello() == "Hello, world!"

    add, x, y = fn . int . int . int
    add [x, y] = x + y

    assert add(3, 5) == 8
    assert type(add(1)) == Fn
    assert add (1) (2) == 3

    greet, name = fn . str . str
    greet ["Bob"] = "Long time no see, ol' pal!"
    greet [name] = "Nice to meet you, " + name

    assert greet("John") == "Nice to meet you, John"

    inc, n = fn . int . int
    inc [n] = n + 1
    
    assert inc(0) == 1

    f, x, y = fn . int . int . int
    f [x, y] = add(2 * x, 2 * y) + 1

    assert f(3, 4) == 15

    fib, n = fn . int . int
    fib [0] = 0
    fib [1] = 1
    fib [n] = fib(n - 1) + fib(n - 2)

    assert fib(6) == 8

    alpha, a = fn . int . int
    beta, b = fn . int . int

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

    foldr, f, z, xs = fn ._ (fn . a . b . b) . a . list.of.b . b
    foldr [f, z, []] = z
    foldr [f, z, xs] = f(xs[0], foldr(f, z, xs[1:]))

    assert foldr(add, 0, [1,2,3,4,5]) == 15

    takeby, k, xs, n = fn . int . list.of.a . int . list.of.a
    takeby [k, xs, n] = xs[n::k]

    @fn
    def foo(x: int, y) -> float:
        return x / int(y)
    assert type(foo) == Fn
    assert foo(1, "2") == (1 / int("2"))
    
    assert type(sum@fn) == Fn
    assert (sum@fn)([1,2,3]) == 6

    assert (lambda x: "Hello " + x)@fn ("World") == "Hello World"
    assert (lambda x, y: x + y)@fn (3, 4) == 7

    # edge case we should still allow
    assert (lambda f: f(123))@fn (str) == "123"

    # TODO: add dummy __

    @fn
    def show(val):
        print(val)
        return val

    # map needs to be typed to work
    mapfn = map@(fn ._ (fn . a . b) . list.of.a . b)
    listfn = list@(fn . a . list.of.b)

    assert (([1, 2] * 5)
            | takeby(2)
            | ((lambda xs: sum(xs, []))@fn & listfn & mapfn)
           )([0,1]) == [1,1,1,1,1,2,2,2,2,2]

    # TODO: get this to run
    # ([1, 2] * 20) | takeby | (lambda x: (map@fn) (x) ([1,2]))@fn | (sum@fn)(__, [])
    # should be equal to sum((lambda x: (x[::2], x[1::2]))([1, 2] * 20), [])
