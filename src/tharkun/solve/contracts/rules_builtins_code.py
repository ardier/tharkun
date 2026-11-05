"""Open compilation/evaluation metadata, without admitting a proof transfer or recipe."""
from tharkun.solve.contracts.rules import Rule, register_rule
from tharkun.solve.contracts.types import ALL_PROPERTY_CLASSES, Contract, Effects, RaiseClaim, ReturnSpec

_DOCS = 'https://docs.python.org/3.14/library/functions.html'


@register_rule
class Compile(Rule):
    """Compilation may invoke conversions, AST access, warnings and audit hooks."""

    names = ('compile',)
    SAMPLES = {'compile': 'x = compile("1", "<source>", "eval")\n'}
    TWINS = {'compile': 'x = compile()\n'}
    CONTRACT = Contract(
        qualname='compile',
        raises=tuple(RaiseClaim(exc,
            'invalid binding, unsupported source, failed source-buffer acquisition, invalid filename/mode/'
            'integer-option/AST shape, or an uncaught TypeError from unsuppressed conversion/AST/compilation/'
            'warning/audit callbacks' if exc == 'TypeError' else
            f'an uncaught {exc} from unsuppressed conversion/AST/compilation/warning/audit callbacks',
            tested=True) for exc in ALL_PROPERTY_CLASSES),
        may_raise_other=True,
        returns=ReturnSpec('types.CodeType | ast.AST',
            'AST when flags include PyCF_ONLY_AST (including PyCF_OPTIMIZED_AST); otherwise a code object; '
            'does not execute the compiled body', tested=True),
        effects=Effects(
            io=True, clock_or_env=True, deterministic=False, may_execute_callback=True,
            notes='Reads caller future flags unless dont_inherit is true and process optimization level when '
                  'optimize is -1; compilation warnings can write to stderr. Filename __fspath__, option '
                  '__index__/truth conversion, AST attribute access, warning and audit hooks can execute '
                  'arbitrary code. Failed source-buffer acquisition is replaced by TypeError; release '
                  'callbacks have unraisable exceptions. Not a purity or literal-expansion model.',
        ),
        source=(f'{_DOCS}#compile',
                'CPython v3.14.4 Python/bltinmodule.c::builtin_compile_impl',
                'CPython v3.14.4 Python/clinic/bltinmodule.c.h::builtin_compile',
                'CPython v3.14.4 Python/pythonrun.c::_Py_SourceAsString, Py_CompileStringObject'),
        tested=True,
    )


@register_rule
class Eval(Rule):
    """The result and effects of evaluated code remain open world."""

    names = ('eval',)
    SAMPLES = {'eval': 'x = eval("1")\n'}
    TWINS = {'eval': 'x = eval()\n'}
    CONTRACT = Contract(
        qualname='eval',
        raises=tuple(RaiseClaim(exc,
            'invalid binding, unsupported source, failed source-buffer acquisition, invalid globals/locals, '
            'code with free variables, or an uncaught TypeError from executed code or unsuppressed '
            'compilation/namespace/warning/audit callbacks' if exc == 'TypeError' else
            f'an uncaught {exc} from executed code or unsuppressed compilation/namespace/warning/audit callbacks',
            tested=True) for exc in ALL_PROPERTY_CLASSES),
        may_raise_other=True,
        returns=ReturnSpec('Any',
            'evaluated result; exec-mode code normally returns None; top-level-await code can return a coroutine',
            tested=True),
        effects=Effects(
            mutates_args=('globals', 'locals'), io=True, clock_or_env=True,
            deterministic=False, may_execute_callback=True,
            notes='Reads caller namespaces/compiler state and may insert __builtins__ into globals; compilation '
                  'warnings can write to stderr. Evaluated code and namespace/buffer/warning/audit hooks can '
                  'have arbitrary effects. Source-buffer acquisition failures are replaced by TypeError; '
                  'namespace KeyError can be suppressed by name fallback. Not a purity, termination or '
                  'literal-expansion model.',
        ),
        source=(f'{_DOCS}#eval',
                'CPython v3.14.4 Python/bltinmodule.c::builtin_eval_impl',
                'CPython v3.14.4 Python/pythonrun.c::_Py_SourceAsString'),
        tested=True,
    )
