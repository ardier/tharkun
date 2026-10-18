"""Fixed native tables; no generated programs are executed."""
import re
import ast

import pytest
from tharkun.solve.absint import prove
from tharkun.solve.contracts.rules import Refused, get_rule
from tharkun.sv.symexec_visitor import CallCtx, SymexecVisitor

# qualname, native args, fixed source expression, expected native result.
CASES = [
 ('str.capitalize', ('a B',), '"a B".capitalize()', 'A b'),
 ('str.casefold', ('Straße',), '"Straße".casefold()', 'strasse'),
 ('str.center', ('a',4,'-'), '"a".center(4,"-")', '-a--'),
 ('str.count', ('aba','a',0,3), '"aba".count("a",0,3)', 2),
 ('str.expandtabs', ('a\tb',4), '"a\\tb".expandtabs(4)', 'a   b'),
 ('str.find', ('aba','b',0,3), '"aba".find("b",0,3)', 1),
 ('str.format_map', ('{x:03d}',{'x':4}), '"{x:03d}".format_map({"x":4})', '004'),
 ('str.isalnum', ('Ab1',), '"Ab1".isalnum()', True),
 ('str.isalpha', ('Ab1',), '"Ab1".isalpha()', False),
 ('str.isascii', ('Ab1',), '"Ab1".isascii()', True),
 ('str.isdecimal', ('Ab1',), '"Ab1".isdecimal()', False),
 ('str.isdigit', ('Ab1',), '"Ab1".isdigit()', False),
 ('str.isidentifier', ('Ab1',), '"Ab1".isidentifier()', True),
 ('str.islower', ('Ab1',), '"Ab1".islower()', False),
 ('str.isnumeric', ('Ab1',), '"Ab1".isnumeric()', False),
 ('str.isprintable', ('Ab1',), '"Ab1".isprintable()', True),
 ('str.isspace', ('Ab1',), '"Ab1".isspace()', False),
 ('str.istitle', ('Ab1',), '"Ab1".istitle()', True),
 ('str.isupper', ('Ab1',), '"Ab1".isupper()', False),
 ('str.ljust', ('a',4,'-'), '"a".ljust(4,"-")', 'a---'),
 ('str.lower', ('ABC',), '"ABC".lower()', 'abc'),
 ('str.lstrip', (' a ',), '" a ".lstrip()', 'a '),
 ('str.maketrans', ('ab','xy'), 'str.maketrans("ab","xy")', {97:120,98:121}),
 ('str.partition', ('a:b',':'), '"a:b".partition(":")', ('a',':','b')),
 ('str.removeprefix', ('abc','a'), '"abc".removeprefix("a")', 'bc'),
 ('str.removesuffix', ('abc','c'), '"abc".removesuffix("c")', 'ab'),
 ('str.rfind', ('aba','a',0,3), '"aba".rfind("a",0,3)', 2),
 ('str.rindex', ('aba','a',0,3), '"aba".rindex("a",0,3)', 2),
 ('str.rjust', ('a',4,'-'), '"a".rjust(4,"-")', '---a'),
 ('str.rpartition', ('a:b',':'), '"a:b".rpartition(":")', ('a',':','b')),
 ('str.rsplit', ('a:b:c',':',1), '"a:b:c".rsplit(":",1)', ['a:b','c']),
 ('str.rstrip', (' a ',), '" a ".rstrip()', ' a'),
 ('str.splitlines', ('a\nb',True), '"a\\nb".splitlines(True)', ['a\n','b']),
 ('str.swapcase', ('Ab',), '"Ab".swapcase()', 'aB'),
 ('str.title', ('a b',), '"a b".title()', 'A B'),
 ('str.translate', ('ab',{97:'xy',98:None}), '"ab".translate({97:"xy",98:None})', 'xy'),
 ('str.upper', ('ab',), '"ab".upper()', 'AB'),
 ('str.zfill', ('-3',4), '"-3".zfill(4)', '-003'),
 ('str.__format__', ('ab','>4'), '"ab".__format__(">4")', '  ab'),
 ('str.__contains__', ('abc','b'), '"abc".__contains__("b")', True),
 ('str.join', (',',('a','b')), '",".join(("a","b"))', 'a,b'),
 ('str.startswith', ('abc',('x','a'),0,3), '"abc".startswith(("x","a"),0,3)', True),
 ('str.endswith', ('abc','c',0,3), '"abc".endswith("c",0,3)', True),
 ('str.split', ('a:b',':'), '"a:b".split(":")', ['a','b']),
 ('str.strip', (' a ',), '" a ".strip()', 'a'),
 ('str.replace', ('aba','a','x',1), '"aba".replace("a","x",1)', 'xba'),
 ('str.format', ('{0!r:>4}',3), '"{0!r:>4}".format(3)', '   3'),
 ('str.index', ('abc','b'), '"abc".index("b")', 1),
 ('str.__getitem__', ('abc',1), '"abc"[1]', 'b'),
 ('format', (3,'04d'), 'format(3,"04d")', '0003'),
 ('repr', ('é',), 'repr("é")', "'é'"),
 ('ascii', ('é',), 'ascii("é")', "'\\xe9'"),
 ('str', (None,), 'str(None)', 'None'),
 ('str.__mod__', ('%#08x',31), '"%#08x" % 31', '0x00001f'),
 ('str.__mod__', ('%*.*f',(8,2,1.5)), '"%*.*f" % (8,2,1.5)', '    1.50'),
 ('str.__mod__', ('%(n)+05d',{'n':3}), '"%(n)+05d" % {"n":3}', '+0003'),
 ('str.__mod__', ('%s %r %a',('é','é','é')), '"%s %r %a" % ("é","é","é")', "é 'é' '\\xe9'"),
 ('syntax.FormattedValue', (3,'>4',114), 'f"{3!r:>4}"', '   3'),
 ('syntax.JoinedStr', ('a','3','b'), 'f"a{3}b"', 'a3b'),
]


@pytest.mark.parametrize('name,args,expression,expected', CASES)
def test_native_and_both_engine_visits(name,args,expression,expected):
    rule = get_rule(name)
    assert rule.native(*args) == expected
    result = rule.accept(SymexecVisitor(None), CallCtx(None,args=args,kwargs={}))
    assert not isinstance(result,Refused), result
    assert result == expected
    # Fixed expressions from the table are interpreted by absint, never exec'd.
    if type(expected) in (str,list,tuple,dict):
        suffix = f'assert len(x) == {len(expected)}\n'
    else:
        suffix = f'assert x == {expected!r}\n'
    verdict = prove('x = '+expression+'\n'+suffix, ('TypeError','AttributeError','AssertionError'))
    assert verdict.verdict == 'TRUE', (name, verdict.explain())


@pytest.mark.parametrize('conversion,value,expected', [
 ('d',-3,'-3'),('i',3,'3'),('o',8,'10'),('u',3,'3'),('x',31,'1f'),('X',31,'1F'),
 ('e',1.5,'1.500000e+00'),('E',1.5,'1.500000E+00'),('f',1.5,'1.500000'),('F',1.5,'1.500000'),
 ('g',1.5,'1.5'),('G',1.5,'1.5'),('c',65,'A'),('s',None,'None'),('r','a',"'a'"),('a','é',"'\\xe9'"),
])
def test_every_percent_conversion(conversion,value,expected):
    assert get_rule('str.__mod__').native('%'+conversion,value) == expected


@pytest.mark.parametrize('name,args,exc', [
 ('str.__mod__',('%d','x'),TypeError),('str.__mod__',('%s %s',(1,)),TypeError),
 ('str.__mod__',('%(x)s',{}),KeyError),('str.__mod__',('%q',3),ValueError),
 ('str.format',('{2}',1),IndexError),('str.format',('{x}',),KeyError),
 ('str.format',('{0!q}',1),ValueError),('str.format_map',('{x}',{}),KeyError),
 ('str.format_map',('{}',{}),ValueError),('format',(3,'q'),ValueError),('format',(3,5),TypeError),
 ('str.center',('x',3,'xx'),TypeError),('str.partition',('x',''),ValueError),
 ('str.rindex',('abc','z'),ValueError),('str.translate',('x',{120:[]}),TypeError),
 ('str.__getitem__',('x',4),IndexError),('str.maketrans',('a','xy'),ValueError),
])
def test_native_error_conditions(name,args,exc):
    with pytest.raises(exc):
        get_rule(name).native(*args)


REGEX = [
 ('re.compile',('a',),re.Pattern),('re.match',('a','a'),re.Match),
 ('re.search',('a','ba'),re.Match),('re.fullmatch',('a','a'),re.Match),
 ('re.findall',('(a)','aba'),['a','a']),('re.finditer',('a','aba'),'iterator'),
 ('re.split',(':','a:b'),['a','b']),('re.sub',('a','x','aba'),'xbx'),
 ('re.subn',('a','x','aba'),('xbx',2)),('re.escape',('a.b',),'a\\.b'),('re.purge',(),None),
 *[('re.Pattern.'+n, (re.compile('(a)'),*args),expected) for n,args,expected in [
  ('match',('a',),re.Match),('search',('ba',),re.Match),('fullmatch',('a',),re.Match),
  ('findall',('aba',),['a','a']),('finditer',('aba',),'iterator'),('split',('aba',),['','a','b','a','']),
  ('sub',('x','aba'),'xbx'),('subn',('x','aba'),('xbx',2)),('scanner',('a',),'scanner')]],
 *[('re.Match.'+n,(re.match('(?P<a>a)','a'),*args),expected) for n,args,expected in [
  ('group',(1,),'a'),('groups',(),('a',)),('groupdict',(),{'a':'a'}),
  ('start',(),0),('end',(),1),('span',(),(0,1)),('expand',(r'\g<a>!',),'a!'),('__getitem__',(1,),'a')]],
]


@pytest.mark.parametrize('name,args,expected',REGEX)
def test_every_regex_sibling(name,args,expected):
    rule = get_rule(name)
    native = rule.native(*args)
    modeled = rule.accept(SymexecVisitor(None), CallCtx(None,args=args,kwargs={}))
    if expected != 'scanner':
        assert not isinstance(modeled,Refused), modeled
    if expected == 'scanner':
        assert native.match().group() == 'a'
        assert isinstance(rule.accept(SymexecVisitor(None),CallCtx(None,args=args)),Refused)
    elif expected == 'iterator':
        assert [m.span() for m in native] == [(0,1),(2,3)]
        assert [m.span() for m in modeled.take()] == [(0,1),(2,3)]
    elif expected in (re.Match,re.Pattern):
        assert type(native) is expected
        if expected is re.Match:
            assert modeled.span() == native.span() and modeled.groups() == native.groups()
        else:
            assert modeled.pattern == native.pattern and modeled.flags == native.flags
    else:
        assert native == modeled == expected


# Regression programs are authored explicitly, including exception-before-later-site cases.
PROGRAMS = [
 ('id = 1000\nx = "Bad task id %d" % id\nassert len(x) > 0\n','TRUE'),
 ('import _sv_verifier as v\nid = v.nondet_int()\nx = "Bad task id %d" % id\n','TRUE'),
 ('x = f"{None!r:>6}"\nassert x == "  None"\n','TRUE'),
 ('x = f"{3:{4}d}"\nassert len(x) == 4\n','TRUE'),
 ('import re\np = re.compile("(?P<x>é)",re.I)\nm = p.fullmatch("É")\nassert m.group("x") == "É"\n','TRUE'),
 ('x = "%d" % "bad"\n','UNKNOWN'),
 ('x = "{missing}".format_map({})\n','UNKNOWN'),
 ('import re\np = re.compile("a")\nx = p.match("a",pos="bad")\n','UNKNOWN'),
]


@pytest.mark.parametrize('source,expected',PROGRAMS)
def test_fixed_regression_programs(source,expected):
    result = prove(source,('TypeError','AttributeError','KeyError','AssertionError'))
    assert result.verdict == expected, result.explain()


@pytest.mark.parametrize('name,args,expression,expected', CASES)
def test_symbolic_executor_dispatch(name,args,expression,expected):
    from tharkun.sv._symexec import _Exec
    executor = _Exec({"re": re, "str": str}, {}, {})
    result = executor.ev(ast.parse(expression, mode="eval").body)
    assert result == expected


def test_symbolic_prefix_bounds_match_native_fixed_cases():
    import z3
    from tharkun.sv._symexec import _Exec
    value = z3.String('text')
    cases = [('abc','',4,5),('abc','',3,2),('abc','a',-4,9),('abc','c',-1,None),('é😀','😀',1,9)]
    for method in ('startswith','endswith'):
        for text,prefix,start,end in cases:
            result = get_rule('str.'+method).accept(SymexecVisitor(_Exec({}, {}, {})),CallCtx(None,args=(value,prefix,start,end)))
            native = getattr(text,method)(prefix,start,end)
            solver = z3.Solver();solver.add(value == text, result != native)
            assert solver.check() == z3.unsat


def test_full_string_method_inventory_is_rule_owned():
    for name in dir(str):
        if not name.startswith('_') and callable(getattr(str,name)):
            assert get_rule('str.'+name) is not None, name


@pytest.mark.parametrize("source",[
 'import re\nit=re.finditer("a","a")\nit.append(1)\n',
 'import re\nit=re.finditer("a","a")\nx=len(it)\n',
 'import _sv_verifier as v\nx = format(v.nondet_int(), 1 if v.nondet_bool() else "d")\n',
 'import _sv_verifier as v\nx = format(v.nondet_int(),"d",3)\n',
 'x = "%1000000000d" % 1\n',
 'import _sv_verifier as v\nx=v.nondet_str()\nx.isalpha(1)\n',
])
def test_fixed_invalid_or_excessive_calls_never_prove(source):
    r=prove(source,("TypeError","AttributeError"))
    assert r.verdict != "TRUE",r.explain()


def test_string_slice_goes_through_runner_rule():
    from tests.contracts.test_engine_audit_consumed import _solve
    got,_=_solve('x="abcdef"[::-2]\n','x == "fdb"')
    assert any(status == 'sat' for status,_ in got.values()),got


@pytest.mark.parametrize("source", [
 'import re\nimport _sv_verifier as v\np=re.compile("(a)",v.nondet_int())\nm=p.match(v.nondet_str())\nif m is not None:\n m.group(2)\n',
 'import _sv_verifier as v\nx=("a" if v.nondet_bool() else None)\nx.upper()\n',
])
def test_unknown_flags_and_mixed_receivers_never_prove(source):
    result=prove(source,("TypeError","AttributeError","IndexError"))
    assert result.verdict != "TRUE",result.explain()


def test_runner_regex_calls_use_evaluated_native_context():
    import z3
    from tests.contracts import _engine_audit as A
    runner,_=A.symexec_harness('import re\nx=re.subn("a","b","aba")\n', extra='if x[1] == 2 and n == 3:\n raise IndexError("hit")\n')
    assert not runner.skipped, runner.skipped
    assert runner.env["x"] == re.subn("a","b","aba")
    assert any(z3.Solver().check(*goals) == z3.sat for goals in runner.goals.values())



def test_symbolic_integer_format_matches_configured_decimal_limit():
    import z3, sys
    from tests.contracts import _engine_audit as A
    runner,_=A.symexec_harness('import _sv_verifier as v\nx=format(v.nondet_int())\n')
    assert not runner.skipped, runner.skipped
    # The decimal-string value is an exact Z3 string on every admitted native path.
    assert any(z3.Solver().check(*goals) == z3.sat for goals in runner.goals.values())


@pytest.mark.parametrize("name,args,kwargs,exc", [
 ("re.Match.groups",(re.match("(a)","a"),1,2),{},TypeError),
 ("re.Match.groupdict",(re.match("(a)","a"),),{"bogus":1},TypeError),
 ("re.Match.expand",(re.match("(a)","a"),r"\g<missing>"),{},IndexError),
 ("re.subn",("(a)",r"\g<missing>","a"),{},IndexError),
 ("re.Pattern.sub",(re.compile("(a)"),r"\g<missing>","a"),{},IndexError),
 ("re.Pattern.subn",(re.compile("(a)"),r"\g<missing>","a"),{},IndexError),
 ("str.maketrans",({"ab":1},),{},ValueError),
 ("str.maketrans",({None:1},),{},TypeError),
 ("str.translate",("x",{120:0x110000}),{},ValueError),
 ("str.__format__",("a","d"),{},ValueError),
 ("str.format",("{} {0}",1),{},ValueError),
 ("str.format",("{0[x]}",{}),{},KeyError),
 ("str.format",("{0.real}",1),{},None),
 ("format",(None,"x"),{},TypeError),
 ("re.compile",("a",re.LOCALE),{},ValueError),
 ("re.compile",(re.compile("a"),re.I),{},ValueError),
 ("re.match",("a",b"a"),{},TypeError),
 ("re.compile",("[",),{},re.PatternError),
])
def test_fixed_error_boundaries_in_native_and_visitors(name,args,kwargs,exc):
    rule=get_rule(name)
    if exc is None:
        assert rule.native(*args,**kwargs) == "1"
        return
    with pytest.raises(exc): rule.native(*args,**kwargs)
    assert isinstance(rule.accept(SymexecVisitor(None),CallCtx(None,args=args,kwargs=kwargs)),Refused)


@pytest.mark.parametrize("source", [
 'import re\nm=re.match("(a)","a")\nm.groups(1,2)\n',
 'import re\nm=re.match("(a)","a")\nm.groupdict(bogus=1)\n',
 'import re\nx=re.subn("(a)",r"\\g<missing>","a")\n',
])
def test_native_regex_shape_and_group_errors_never_prove(source):
    result=prove(source,("TypeError","IndexError"))
    assert result.verdict != "TRUE",result.explain()


INDEX_CALLS = [
 ("str.center",("a",)), ("str.count",("aba","a")),
 ("str.expandtabs",("a\tb",)), ("str.find",("abc","b")),
 ("str.ljust",("a",)), ("str.rfind",("abc","b")),
 ("str.rindex",("abc","b")), ("str.rjust",("a",)),
 ("str.rsplit",("a:b",":")),
 ("str.zfill",("1",)),
]

@pytest.mark.parametrize("name,args",INDEX_CALLS)
@pytest.mark.parametrize("exc_name",("TypeError","AttributeError","IndexError","KeyError","AssertionError","ZeroDivisionError","FloatingPointError"))
def test_new_index_argument_callbacks_propagate(name,args,exc_name):
    from tests.contracts._propagation import PROPERTY_CLASSES,raiser,nonproperty_raiser
    rule=get_rule(name)
    with pytest.raises(PROPERTY_CLASSES[exc_name]): rule.native(*args,raiser("__index__",exc_name)())
    with pytest.raises(ValueError): rule.native(*args,nonproperty_raiser("__index__",ValueError)())
    assert rule.contract.effects.may_execute_callback and rule.contract.may_raise_other


@pytest.mark.parametrize("name,args,hook", [
 ("str.format_map",("{x}",),"__getitem__"),
 ("str.translate",("x",),"__getitem__"),
])
@pytest.mark.parametrize("exc_name",("TypeError","AttributeError","AssertionError","ZeroDivisionError","FloatingPointError"))
def test_new_mapping_argument_callbacks_propagate(name,args,hook,exc_name):
    from tests.contracts._propagation import PROPERTY_CLASSES,raiser
    with pytest.raises(PROPERTY_CLASSES[exc_name]): get_rule(name).native(*args,raiser(hook,exc_name)())


def test_translate_swallows_only_lookup_errors():
    from tests.contracts._propagation import raiser
    assert get_rule("str.translate").native("x",raiser("__getitem__","KeyError")()) == "x"
    assert get_rule("str.translate").native("x",raiser("__getitem__","IndexError")()) == "x"


def test_regex_and_string_sibling_inventories():
    for owner,prefix in ((str,"str."),(re.Pattern,"re.Pattern."),(re.Match,"re.Match.")):
        for name in dir(owner):
            if not name.startswith("_") and callable(getattr(owner,name)):
                assert get_rule(prefix+name) is not None,(prefix,name)


@pytest.mark.parametrize("exc_name",("TypeError","AttributeError","IndexError","KeyError","AssertionError","ZeroDivisionError","FloatingPointError"))
def test_splitlines_truth_callback_propagates(exc_name):
    from tests.contracts._propagation import PROPERTY_CLASSES,raiser
    with pytest.raises(PROPERTY_CLASSES[exc_name]): get_rule("str.splitlines").native("a\nb",raiser("__bool__",exc_name)())
    assert get_rule("str.splitlines").native("a\nb",1.5) == ["a\n","b"]


def test_native_codec_arguments_and_recursive_values_refuse_without_crashing():
    for args in (("abc",[]),("abc","utf-8",[])):
        assert isinstance(get_rule("str.encode").accept(SymexecVisitor(None),CallCtx(None,args=args)),Refused)
    cyclic=[];cyclic.append(cyclic)
    assert isinstance(get_rule("repr").accept(SymexecVisitor(None),CallCtx(None,args=(cyclic,))),Refused)


@pytest.mark.parametrize("name,args", [
 ("re.search",("(a+)+$","a"*30+"!")),
 ("re.search",("(a|aa)+$","a"*30+"!")),
 ("re.search",("(a)\\1","aa")),
 ("re.compile",("a",re.DEBUG)),
])
def test_regex_resource_refusals_are_named_in_both_engines(name,args):
    result=get_rule(name).accept(SymexecVisitor(None),CallCtx(None,args=args))
    assert isinstance(result,Refused) and result.qualname == name,result


@pytest.mark.parametrize("source", [
 'import re\nx=re.search("(a+)+$","aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa!")\n',
 'import re\nx=re.compile("a",re.DEBUG)\n',
])
def test_regex_resource_refusals_reach_absint(source):
    result=prove(source,("TypeError",))
    assert result.verdict == "UNKNOWN" and "not modelled" in result.reason,result.explain()


@pytest.mark.parametrize("pattern,value,expected", [
 ("%-6d",-3,"-3    "), ("% d",3," 3"), ("%.2s","abcd","ab"),
 ("%.*s",(2,"abcd"),"ab"), ("%*d",(-4,3),"3   "),
 ("%hd %ld %Ld",(1,2,3),"1 2 3"), ("%%",(),"%"),
 ("%(a(b)c)s",{"a(b)c":"x"},"x"), ("%s",[1,2],"[1, 2]"),
 ("%c",0x10ffff,"\U0010ffff"),
])
def test_remaining_percent_flags_and_operand_shapes(pattern,value,expected):
    rule=get_rule("str.__mod__")
    assert rule.native(pattern,value) == expected
    assert rule.accept(SymexecVisitor(None),CallCtx(None,args=(pattern,value))) == expected


@pytest.mark.parametrize("source,expected", [
 ('from re import subn as replace\nx=replace("a","b","aba")\n', ("bbb",2)),
 ('from builtins import format as fmt\nx=fmt(3,"04d")\n', "0003"),
 ('x=str.upper("abc")\n', "ABC"),
])
def test_fixed_native_alias_and_unbound_entrypoints(source,expected):
    from tests.contracts import _engine_audit as A
    runner,_=A.symexec_harness(source)
    assert not runner.skipped,runner.skipped
    assert runner.env["x"] == expected
    assert prove(source,("TypeError","AttributeError")).verdict == "TRUE"


def test_regex_flag_enum_uses_native_semantics():
    rule=get_rule("re.match")
    result=rule.accept(SymexecVisitor(None),CallCtx(None,args=("é","É",re.I)))
    assert result.group() == "É"



def test_string_membership_operator_uses_the_shared_rule():
    import z3
    from tests.contracts.test_engine_audit_consumed import _solve
    source='x = "b" in "abc"\nassert x\n'
    assert prove(source,("TypeError","AssertionError")).verdict == "TRUE"
    got,_=_solve('x="b" in "abc"\n','x')
    assert any(status == "sat" for status,_ in got.values()),got
    value=z3.String("contains_text")
    rule=get_rule("str.__contains__")
    result=rule.accept(SymexecVisitor(None),CallCtx(None,args=(value,"é")))
    solver=z3.Solver();solver.add(value=="é😀",z3.Not(result))
    assert solver.check() == z3.unsat


@pytest.mark.parametrize("name,args,expected", [
 ("re.Match.group",(re.match("(a)","a"),0,1),("a","a")),
 ("re.Match.groupdict",(re.match("(?P<x>a)?",""),7),{"x":7}),
 ("re.Match.expand",(re.match(b"(a)",b"a"),b"\\1!"),b"a!"),
 ("re.Pattern.subn",(re.compile(b"a"),b"b",b"aba"),(b"bbb",2)),
 ("re.Pattern.split",(re.compile("(a)?b"),"b"),["",None,""]),
])
def test_regex_result_kinds_defaults_and_missing_captures(name,args,expected):
    rule=get_rule(name)
    assert rule.native(*args) == expected
    assert rule.accept(SymexecVisitor(None),CallCtx(None,args=args)) == expected


def test_purge_wrong_shape_is_native_typeerror():
    with pytest.raises(TypeError): get_rule("re.purge").native(1)



def test_cpython314_conversion_precedes_nested_spec_evaluation():
    log=[]
    class Value:
        def __repr__(self):
            log.append("conversion")
            return "x"
    def spec():
        log.append("spec")
        return ">2"
    assert f"{Value()!r:{spec()}}" == " x"
    assert log == ["conversion","spec"]


def test_symbolic_conversion_failure_precedes_spec_failure():
    import sys
    from tharkun.sv._symexec import _Exec,Unsupported
    digits=sys.get_int_max_str_digits()
    if not digits: pytest.skip("interpreter decimal conversion cap disabled")
    executor=_Exec({}, {"value":10**digits}, {})
    with pytest.raises(Unsupported,match="str: native ValueError"):
        executor.ev(ast.parse('f"{value!s:{1 // 0}}"',mode="eval").body)


def test_absint_conversion_failure_skips_nested_spec():
    import sys
    from tharkun.solve.absint.interp import Analyzer,FuncCtx
    from tharkun.solve.absint.aval import const_int
    from tharkun.solve.absint.state import Frame,State
    digits=sys.get_int_max_str_digits()
    if not digits: pytest.skip("interpreter decimal conversion cap disabled")
    value=10**digits
    with pytest.raises(ValueError):
        f"{value!s:{1 // 0}}"
    # Seed the exact abstract value; the arithmetic domain widens very large shifts.
    analyzer=Analyzer('x=f"{value!s:{1 // 0}}"\n')
    state=State();state.frames.append(Frame("g"))
    state.frame.env["value"]=const_int(value)
    context=FuncCtx(analyzer.tree,{"value","x"},set(),is_module=True)
    analyzer.module_ctx=context
    result=analyzer.eval(analyzer.tree.body[0].value,state,context)
    assert state.bottom and result.is_bot()
    assert not any(check.op == "divide" for check in analyzer.checks.values())


def test_str_bytes_constructor_codec_guard_is_shared():
    for args in ((b"a","custom"),(b"a",[])):
        result=get_rule("str").accept(SymexecVisitor(None),CallCtx(None,args=args))
        assert isinstance(result,Refused),result


@pytest.mark.parametrize("name,error",[("str.__mod__","TypeError"),("syntax.JoinedStr","TypeError"),("syntax.FormattedValue","KeyError")])
def test_nonfinite_format_call_shape_keeps_native_errors_unproven(name,error):
    from tharkun.solve.absint.interp import Analyzer
    from tharkun.solve.absint.aval import ANYSTR,const_int,const_str,int_iv
    from tharkun.solve.absint.state import State
    from tharkun.solve.absint.visitor import AbsintVisitor,CallCtx as AbstractCallCtx
    analyzer=Analyzer("x=0\n");state=State();node=analyzer.tree.body[0]
    if name == "str.__mod__":
        abstract_args=(const_str("%d"),int_iv(None,None));native_args=("%d",3);kwargs={"bad":const_int(1)};native_kwargs={"bad":1}
    elif name == "syntax.JoinedStr":
        abstract_args=(ANYSTR,);native_args=("a",);kwargs={"bad":const_int(1)};native_kwargs={"bad":1}
    else:
        abstract_args=(int_iv(None,None),const_str(""),const_int(999));native_args=(3,"",999);kwargs={};native_kwargs={}
    rule=get_rule(name)
    with pytest.raises({"TypeError":TypeError,"KeyError":KeyError}[error]):
        rule.native(*native_args,**native_kwargs)
    result=rule.accept(AbsintVisitor(analyzer),AbstractCallCtx(state,abstract_args,kwargs,node))
    if not isinstance(result,Refused):
        check=next(check for check in analyzer.checks.values() if check.op == name+":"+error)
        assert not check.ok,check


def test_symbolic_fstring_conversion_never_hashes_unverified_objects():
    import z3
    class Unverified:
        def __hash__(self):
            raise AssertionError("target hash executed")
    for conversion in ([],Unverified()):
        result=get_rule("syntax.FormattedValue").accept(SymexecVisitor(None),CallCtx(None,args=(z3.String("format_value"),"",conversion)))
        assert isinstance(result,Refused),result


def test_fixed_high_unicode_symbolic_string_operations():
    import z3
    text=z3.String('unicode_text')
    value='\U0010ffffx'
    for name,part,expected in [('str.startswith','\U0010ffff',True),('str.endswith','x',True),('str.__contains__','\U0010ffff',True)]:
        assert get_rule(name).native(value,part) == expected
        result=get_rule(name).accept(SymexecVisitor(None),CallCtx(None,args=(text,part)))
        if isinstance(result,Refused):
            assert result.qualname == name
        else:
            solver=z3.Solver();solver.add(text==value,result!=expected)
            assert solver.check()==z3.unsat,(name,result,solver)
            solver=z3.Solver();solver.add(text==value,result==expected)
            assert solver.check()==z3.sat,(name,result,solver)


@pytest.mark.parametrize("source",[
 'import re\nassert str(re.I) == "2"\n',
 'import re\nassert "%s" % re.I == "2"\n',
 'import re\nassert f"{re.I!s}" == "2"\n',
 'import re\nassert ~re.I == -3\n',
])
def test_regexflag_identity_never_proves_plain_int_semantics(source):
    assert str(re.I) == "re.IGNORECASE"
    assert "%s" % re.I == "re.IGNORECASE"
    assert f"{re.I!s}" == "re.IGNORECASE"
    assert ~re.I != -3
    result=prove(source,("AssertionError",))
    assert result.verdict != "TRUE",result.explain()


@pytest.mark.parametrize("source",[
 'import re\nx=re.I+re.M\nassert x==10\n',
 'import re\nx=-re.I\nassert x==-2\n',
 'import re\nx=re.search("a","A",re.I|re.M)\nassert x.group()=="A"\n',
])
def test_fixed_regexflag_arithmetic_and_pattern_entrypoints(source):
    assert re.I+re.M == 10
    assert -re.I == -2
    assert re.search("a","A",re.I|re.M).group() == "A"
    result=prove(source,("TypeError","AttributeError","AssertionError"))
    assert result.verdict == "TRUE",result.explain()


@pytest.mark.parametrize("source",[
 'from re import I as flag\nassert str(flag)=="2"\n',
 'import re\nassert bool(re.NOFLAG)\n',
])
def test_fixed_regexflag_alias_and_zero_never_prove_wrong_assertions(source):
    assert str(re.I) != "2" and not bool(re.NOFLAG)
    result=prove(source,("AssertionError",))
    assert result.verdict != "TRUE",result.explain()


@pytest.mark.parametrize("expression,expected",[
 ('str(+flag)',"2"), ('repr(+flag)',"2"), ('f"{+flag!s}"',"2"),
])
def test_symbolic_unary_plus_removes_regexflag_enum_identity(expression,expected):
    from tharkun.sv._symexec import _Exec
    assert type(+re.I) is int and str(+re.I) == "2"
    executor=_Exec({"flag":re.I}, {}, {})
    assert executor.ev(ast.parse(expression,mode="eval").body) == expected


@pytest.mark.parametrize("expression,expected",[
 ('str(re.I)',str(re.I)), ('repr(re.I)',repr(re.I)),
 ('f"{re.I!s}"',f"{re.I!s}"), ('str(re.I|re.M)',str(re.I|re.M)),
])
def test_symbolic_regexflag_module_constants_keep_enum_identity(expression,expected):
    from tharkun.sv._symexec import _Exec
    executor=_Exec({"re":re}, {}, {})
    assert executor.ev(ast.parse(expression,mode="eval").body) == expected


def test_symbolic_regexflag_bitwise_result_refuses_without_enum_encoding():
    import z3
    from tharkun.sv._symexec import _Exec,Unsupported
    mask=re.RegexFlag(3)
    assert type(5 & mask) is re.RegexFlag
    executor=_Exec({"mask":mask}, {"draw":z3.Int("flag_draw")}, {})
    with pytest.raises(Unsupported,match="re.RegexFlag"):
        executor.ev(ast.parse('draw & mask',mode="eval").body)


# [internal ref] / MG151: fixed native controls for joined percent operands and attribute reads.
@pytest.mark.parametrize("source,template,bad_operand", [
 ('import _sv_verifier\nr=(1,2) if _sv_verifier.nondet_int()>0 else 5\nx="%d%d" % r\n', '%d%d', 5),
 ('import _sv_verifier\ndef render(r):\n    return "%d%d" % r\nr=(1,2) if _sv_verifier.nondet_int()>0 else 5\nx=render(r)\n', '%d%d', 5),
 ('import _sv_verifier\nr=() if _sv_verifier.nondet_int()>0 else 5\nx="" % r\n', '', 5),
 ('import _sv_verifier\nr=(1,2) if _sv_verifier.nondet_int()>0 else "5"\nx="%d%d" % r\n', '%d%d', '5'),
 ('import _sv_verifier\nr=(1,2) if _sv_verifier.nondet_int()>0 else None\nx="%d%d" % r\n', '%d%d', None),
 ('import _sv_verifier\nr=(1,2) if _sv_verifier.nondet_int()>0 else 5.0\nx="%d%d" % r\n', '%d%d', 5.0),
 ('import _sv_verifier\nr=(1,2) if _sv_verifier.nondet_int()>0 else True\nx="%d%d" % r\n', '%d%d', True),
 ('import re, _sv_verifier\nr=(1,2) if _sv_verifier.nondet_int()>0 else re.I\nx="%d%d" % r\n', '%d%d', re.I),
])
def test_percent_tuple_scalar_join_never_proves_missing_arguments(source, template, bad_operand):
    with pytest.raises(TypeError):
        template % bad_operand
    assert '%d%d' % (1,2) == '12' and '' % () == ''
    result=prove(source, ('TypeError','AttributeError'))
    assert result.verdict != 'TRUE', result.explain()


@pytest.mark.parametrize("source,native_value,attribute", [
 ('import re\nx=re.compile("a").nope\n', re.compile('a'), 'nope'),
 ('import re\nx=re.match("a","a").nope\n', re.match('a','a'), 'nope'),
 ('import re\np=re.compile("a")\nx=p.__nope__\n', re.compile('a'), '__nope__'),
 ('import re, _sv_verifier\np=re.compile(str(_sv_verifier.nondet_int()))\nx=p.nope\n', re.compile('1'), 'nope'),
 ('import re, _sv_verifier\np=re.compile(".")\nm=p.match(str(_sv_verifier.nondet_int()))\nif m is not None:\n    x=m.nope\n', re.match('.', '1'), 'nope'),
 ('import re\nx=re.compile("a").string\n', re.compile('a'), 'string'),
 ('import re\nx=re.match("a","a").pattern\n', re.match('a','a'), 'pattern'),
])
def test_regex_attribute_read_never_invents_an_existing_method(source,native_value,attribute):
    with pytest.raises(AttributeError):
        getattr(native_value,attribute)
    result=prove(source, ('TypeError','AttributeError'))
    assert result.verdict != 'TRUE', result.explain()


@pytest.mark.parametrize('tag', [
 ('abstract-pattern',None), ('abstract-match',re.compile('a')),
 ('native-regex',re.compile('a')), ('native-regex',re.match('a','a')),
 ('native-flag',re.I), ('native-slice',slice(1,2)), ('type','str'),
 ('native-method','re.Pattern.match',('native-regex',re.compile('a'))),
])
def test_native_tag_attribute_admission_siblings(tag):
    from tharkun.solve.absint.interp import Analyzer,Unsupported
    from tharkun.solve.absint.state import State
    analyzer=Analyzer('x=0\n')
    with pytest.raises(Unsupported):
        analyzer.static_attr(State(),tag,'__nope__',analyzer.tree.body[0])


@pytest.mark.parametrize('source', [
 'import _sv_verifier\nr=(_sv_verifier.nondet_int(),2)\nx="%d%d" % r\n',
 'import re\np=re.compile("a")\nf=p.match\nx=p.pattern\nn=p.groups\n',
 'import re\nm=re.match("a","a")\nf=m.group\nx=m.string\ng=m.groups\nxs=g()\n',
])
def test_review1153_valid_native_operations_remain_admitted(source):
    assert '%d%d' % (0,2) == '02'
    assert callable(re.compile('a').match) and re.compile('a').pattern == 'a'
    assert callable(re.match('a','a').group) and re.match('a','a').string == 'a'
    result=prove(source, ('TypeError','AttributeError'))
    assert result.verdict == 'TRUE', result.explain()


@pytest.mark.parametrize('bound,source', [
    ([1], "x = 'ab'[[1]:]"),
    ({}, "x = 'ab'[:{}]"),
    (([1],), "x = 'ab'[([1],):]"),
])
def test_unhashable_native_slice_bounds_are_a_certain_type_error_without_internal_crash(bound, source):
    """A list, dict or tuple-with-a-list bound has no hash, so no ``("native-slice", ...)`` tag can hold it (``Analyzer.subscript_load``): it
    is a ``("bad-slice",)`` key ([internal ref]), a certain ``TypeError``: recorded, unproven, and the path ends -- never an internal crash, never TRUE."""
    with pytest.raises(TypeError):
        'ab'[slice(bound, None)]
    result = prove(source, ('TypeError',))
    assert result.verdict != 'TRUE', result.explain()
    assert 'TypeError' in result.explain() and 'str.__getitem__' in result.explain()
    assert prove(source + "\nc = 1 // 0\n", ('ZeroDivisionError',)).verdict == 'TRUE'    # the sentinel after it is never reached


@pytest.mark.parametrize('name,make_args', [
    ('format', lambda obj: (obj, '')),
    ('str', lambda obj: (obj,)),
    ('repr', lambda obj: (obj,)),
    ('ascii', lambda obj: (obj,)),
    ('syntax.FormattedValue', lambda obj: (obj, '', -1)),
    ('str.format', lambda obj: ('{}', obj)),
])
def test_native_string_visits_preserve_internal_object_refusal(name, make_args):
    from tharkun.sv._symexec import Ref
    class A:
        def __format__(self, spec):
            raise AssertionError('target hook must not execute')
        def __str__(self):
            raise AssertionError('target hook must not execute')
        def __repr__(self):
            raise AssertionError('target hook must not execute')
    obj = Ref(1, A, frozenset({'__format__', '__str__', '__repr__'}))
    result = get_rule(name).accept(SymexecVisitor(None), CallCtx(None, args=make_args(obj), kwargs={}))
    assert isinstance(result, Refused)
    assert 'applied to a A object' in str(result)


@pytest.mark.parametrize('name,make_args', [
    ('format', lambda obj: (obj, '')),
    ('str', lambda obj: (obj,)),
    ('str.__contains__', lambda obj: ('ab', obj)),
    ('str.startswith', lambda obj: (obj, 'a')),
    ('str.endswith', lambda obj: (obj, 'b')),
    ('str.startswith', lambda obj: (__import__('z3').String('s'), 'a', obj)),
    ('str.endswith', lambda obj: (__import__('z3').String('s'), 'b', obj)),
])
def test_ref_admission_does_not_read_target_class(name, make_args):
    class Hostile:
        def __getattribute__(self, name):
            raise AssertionError('target attribute lookup: ' + name)
    result = get_rule(name).accept(
        SymexecVisitor(None), CallCtx(None, args=make_args(Hostile()), kwargs={}))
    assert isinstance(result, Refused)


def test_closed_native_admission_never_hashes_or_inspects_target_types():
    armed = False
    class Meta(type):
        def __hash__(cls):
            raise AssertionError("target metaclass hash")
        def __eq__(cls, other):
            raise AssertionError("target metaclass equality")
        def __getattribute__(cls, name):
            if armed:
                raise AssertionError("target metaclass lookup: " + name)
            return type.__getattribute__(cls, name)
    class Hostile(metaclass=Meta):
        pass
    obj = Hostile()
    armed = True
    result = get_rule("format").accept(SymexecVisitor(None), CallCtx(None, args=(obj, ""), kwargs={}))
    assert isinstance(result, Refused)


def test_symbolic_globals_refuse_int_and_enum_subclasses_without_callbacks():
    from enum import IntFlag
    from tharkun.sv._symexec import _Exec, Unsupported

    class Meta(type):
        def __hash__(cls):
            raise AssertionError("target metaclass hash")
        def __eq__(cls, other):
            raise AssertionError("target metaclass equality")

    class HostileInt(int, metaclass=Meta):
        def __pos__(self):
            raise AssertionError("target unary plus")

    class OtherFlag(IntFlag):
        VALUE = 2
        def __pos__(self):
            raise AssertionError("target enum unary plus")

    for value in (HostileInt(2), OtherFlag.VALUE):
        executor = _Exec({"flag": value}, {}, {})
        with pytest.raises(Unsupported, match="name flag"):
            executor.ev(ast.parse("+flag", mode="eval").body)
