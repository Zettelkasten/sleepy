from sleepy.ast import FileAst, AbstractScopeAst, annotate_ast, ExpressionStatementAst, StructDeclarationAst, \
  ReturnStatementAst, AssignStatementAst, IdentifierExpressionAst, MemberExpressionAst, \
  BinaryOperatorExpressionAst, IfStatementAst, WhileStatementAst, UnaryOperatorExpressionAst, ConstantExpressionAst, \
  StringLiteralExpressionAst, CallExpressionAst, AnnotationAst, UnionTypeAst, IdentifierTypeAst, ReferenceExpressionAst

from sleepy.ast_value_parsing import parse_assign_op, parse_long, parse_double, parse_float, parse_char, parse_string, \
  parse_hex_int
from sleepy.functions import FunctionDeclarationAst
from sleepy.grammar import AttributeGrammar, Production
from sleepy.parser import ParserGenerator
from sleepy.builtin_symbols import SLEEPY_DOUBLE, SLEEPY_FLOAT, SLEEPY_INT, SLEEPY_LONG, SLEEPY_CHAR

SLEEPY_ATTR_GRAMMAR = AttributeGrammar.from_dict(
  prods_attr_rules={
    Production('TopLevelStmt', 'StmtList'): {
      'ast': lambda _pos, stmt_list: FileAst(_pos, stmt_list=stmt_list(1))
    },

    Production(';?') : {},
    Production(';?', ';') : {},

    Production('separator', ';'): {},
    Production('separator', 'new_line'): {},
    Production('separator?'): {},
    Production('separator?', 'separator'): {},
    Production('Scope', '{', 'StmtList', '}'): {
      'ast': lambda _pos, stmt_list: AbstractScopeAst(_pos, stmt_list=stmt_list(2))},

    Production('SeparatedStmt', 'Stmt', 'separator') : { 'ast': 'ast.1' },
    Production('SeparatedStmt', 'If1Stmt') : { 'ast': 'ast.1' },
    Production('SeparatedStmt', 'If2Stmt') : { 'ast': 'ast.1' },

    Production('AnnotatedStmt', 'AnnotationList', 'Stmt'): {
      'ast': lambda annotation_list, ast: annotate_ast(ast(2), annotation_list(1))
    },
    Production('AnnotatedSeparatedStmt', 'AnnotationList', 'SeparatedStmt'): {
      'ast': lambda annotation_list, ast: annotate_ast(ast(2), annotation_list(1))
    },

    Production('StmtList'): { 'stmt_list': [] },
    Production('StmtList', 'AnnotatedStmt'): {
      'stmt_list': lambda ast: [ast(1)]
    },
    Production('StmtList', 'AnnotatedSeparatedStmt', 'StmtList'): {
      'stmt_list': lambda ast, stmt_list: [ast(1)] + stmt_list(2)
    },


    Production('If1Stmt', 'if', 'Expr', 'Scope'): {
      'ast': lambda _pos, ast: IfStatementAst(_pos, condition_val=ast(2), true_scope=ast(3), false_scope=None)
    },
    Production('If2Stmt', 'if', 'Expr', 'Scope', 'else', 'Scope'): {
      'ast': lambda _pos, ast: IfStatementAst(_pos, condition_val=ast(2), true_scope=ast(3), false_scope=ast(5))
    },
    Production('Stmt', 'Expr'): {
      'ast': lambda _pos, ast: ExpressionStatementAst(_pos, expr=ast(1))},
    Production('Stmt', 'func', 'identifier', 'TemplateIdentifierList', '(', 'TypedIdentifierList', ')', 'ReturnType', 'Scope'): {  # noqa
      'ast': lambda _pos, identifier, identifier_list, type_list, annotation_list, mutates_list, ast: (
        FunctionDeclarationAst(
          _pos, identifier=identifier(2), templ_identifiers=identifier_list(3), arg_identifiers=identifier_list(5),
          arg_types=type_list(5), arg_annotations=annotation_list(5), arg_mutates=mutates_list(5), return_type=ast(7),
          return_annotation_list=annotation_list(7), body_scope=ast(8)))},
    Production('Stmt', 'func', 'Op', 'TemplateIdentifierList', '(', 'TypedIdentifierList', ')', 'ReturnType', 'Scope'): {  # noqa
      'ast': lambda _pos, op, identifier_list, type_list, annotation_list, mutates_list, ast: (
        FunctionDeclarationAst(
          _pos, identifier=op(2), templ_identifiers=identifier_list(3), arg_identifiers=identifier_list(5),
          arg_types=type_list(5), arg_annotations=annotation_list(5), arg_mutates=mutates_list(5), return_type=ast(7),
          return_annotation_list=annotation_list(7), body_scope=ast(8)))},
    Production('Stmt', 'func', '(', 'AnnotationList', 'Mutates?', 'identifier', ':', 'Type', ')', '[', 'TypedIdentifierList', ']', 'ReturnType', 'Scope'): {  # noqa
      'ast': lambda _pos, identifier, identifier_list, type_list, annotation_list, mutates, mutates_list, ast: (
        FunctionDeclarationAst(
          _pos, identifier='index', arg_identifiers=[identifier(5)] + identifier_list(10), templ_identifiers=[],
          arg_types=[ast(7)] + type_list(10), arg_annotations=[annotation_list(3)] + annotation_list(10),
          arg_mutates=[mutates(4)] + mutates_list(10),
          return_type=ast(12), return_annotation_list=annotation_list(12), body_scope=ast(13)))},
    Production('Stmt', 'extern_func', 'identifier', '(', 'TypedIdentifierList', ')', 'ReturnType'): {
      'ast': lambda _pos, identifier, identifier_list, type_list, annotation_list, mutates_list, ast: (
        FunctionDeclarationAst(
          _pos, identifier=identifier(2), templ_identifiers=[], arg_identifiers=identifier_list(4),
          arg_types=type_list(4), arg_annotations=annotation_list(4), arg_mutates=mutates_list(4), return_type=ast(6),
          return_annotation_list=annotation_list(6), body_scope=None))},
    Production('Stmt', 'struct', 'identifier', 'TemplateIdentifierList', '{', 'MemberList', '}'): {
      'ast': lambda _pos, identifier, identifier_list, type_list, annotation_list: StructDeclarationAst(
        _pos, struct_identifier=identifier(2), templ_identifiers=identifier_list(3),
        member_identifiers=identifier_list(5), member_types=type_list(5), member_annotations=annotation_list(5))},
    Production('Stmt', 'return', 'ExprList'): {
      'ast': lambda _pos, val_list: ReturnStatementAst(_pos, return_exprs=val_list(2))},
    Production('Stmt', 'Expr', ':', 'Type', '=', 'Expr'): {
      'ast': lambda _pos, ast: AssignStatementAst(_pos, var_target=ast(1), var_val=ast(5), declared_var_type=ast(3))},
    # TODO: Handle equality operator in a saner way
    Production('Stmt', 'Expr', '=', 'Expr'): {
      'ast': lambda _pos, ast: (
        AssignStatementAst(_pos, var_target=ast(1), var_val=ast(3), declared_var_type=None)
        if isinstance(ast(1), IdentifierExpressionAst) or isinstance(ast(1), MemberExpressionAst)
        else ExpressionStatementAst(_pos, BinaryOperatorExpressionAst(
          _pos, op='=', left_expr=ast(1), right_expr=ast(3))))},
    Production('Stmt', 'Expr', 'assign_op', 'Expr'): {
      'ast': lambda _pos, ast, op: AssignStatementAst(
        _pos, var_target=ast(1), var_val=BinaryOperatorExpressionAst(
          _pos, op=op(2), left_expr=ast(1), right_expr=ast(3)), declared_var_type=None)},
    Production('Stmt', 'while', 'Expr', 'Scope'): {
      'ast': lambda _pos, ast: WhileStatementAst(_pos, condition_val=ast(2), body_scope=ast(3))},
    Production('Expr', 'Expr', 'cmp_op', 'SumExpr'): {
      'ast': lambda _pos, ast, op: BinaryOperatorExpressionAst(_pos, op=op(2), left_expr=ast(1), right_expr=ast(3))},
    Production('Expr', 'SumExpr'): {
      'ast': 'ast.1'},
    Production('SumExpr', 'SumExpr', 'sum_op', 'ProdExpr'): {
      'ast': lambda _pos, ast, op: BinaryOperatorExpressionAst(_pos, op(2), ast(1), ast(3))},
    Production('SumExpr', 'ProdExpr'): {
      'ast': 'ast.1'},
    Production('ProdExpr', 'ProdExpr', 'prod_op', 'MemberExpr'): {
      'ast': lambda _pos, ast, op: BinaryOperatorExpressionAst(_pos, op(2), ast(1), ast(3))},
    Production('ProdExpr', 'MemberExpr'): {
      'ast': 'ast.1'},
    Production('MemberExpr', 'MemberExpr', '.', 'identifier'): {
      'ast': lambda _pos, ast, identifier: MemberExpressionAst(_pos, ast(1), identifier(3))},
    Production('MemberExpr', 'NegExpr'): {
      'ast': 'ast.1'},
    Production('NegExpr', 'sum_op', 'PrimaryExpr'): {
      'ast': lambda _pos, ast, op: UnaryOperatorExpressionAst(_pos, op(1), ast(2))},
    Production('NegExpr', 'PrimaryExpr'): {
      'ast': 'ast.1'},
    Production('PrimaryExpr', 'int'): {
      'ast': lambda _pos, number: ConstantExpressionAst(_pos, number(1), SLEEPY_INT)},
    Production('PrimaryExpr', 'long'): {
      'ast': lambda _pos, number: ConstantExpressionAst(_pos, number(1), SLEEPY_LONG)},
    Production('PrimaryExpr', 'double'): {
      'ast': lambda _pos, number: ConstantExpressionAst(_pos, number(1), SLEEPY_DOUBLE)},
    Production('PrimaryExpr', 'float'): {
      'ast': lambda _pos, number: ConstantExpressionAst(_pos, number(1), SLEEPY_FLOAT)},
    Production('PrimaryExpr', 'char'): {
      'ast': lambda _pos, number: ConstantExpressionAst(_pos, number(1), SLEEPY_CHAR)},
    Production('PrimaryExpr', 'str'): {
      'ast': lambda _pos, string: StringLiteralExpressionAst(_pos, string(1))},
    Production('PrimaryExpr', 'hex_int'): {
      'ast': lambda _pos, number: ConstantExpressionAst(_pos, number(1), SLEEPY_INT)},
    Production('PrimaryExpr', 'identifier'): {
      'ast': lambda _pos, identifier: IdentifierExpressionAst(_pos, identifier(1))},
    Production('PrimaryExpr', 'PrimaryExpr', '(', 'ExprList', ')'): {
      'ast': lambda _pos, ast, val_list: CallExpressionAst(
        _pos, func_expr=ast(1), func_arg_exprs=val_list(3))},
    Production('PrimaryExpr', 'ref', '(', 'Expr', ')'): {
      'ast': lambda _pos, ast: ReferenceExpressionAst(_pos, arg_expr=ast(3))},
    Production('PrimaryExpr', 'PrimaryExpr', '[', 'ExprList', ']'): {
      'ast': lambda _pos, ast, val_list: CallExpressionAst(
        _pos, func_expr=IdentifierExpressionAst(_pos, identifier='index'), func_arg_exprs=[ast(1)] + val_list(3))},
    Production('PrimaryExpr', '(', 'Expr', ')'): {
      'ast': 'ast.2'},
    Production('AnnotationList'): {
      'annotation_list': []},
    Production('AnnotationList', 'Annotation', 'AnnotationList'): {
      'annotation_list': lambda ast, annotation_list: [ast(1)] + annotation_list(2)},
    Production('Annotation', '@', 'identifier'): {
      'ast': lambda _pos, identifier: AnnotationAst(_pos, identifier(2))},
    Production('IdentifierList'): {
      'identifier_list': []},
    Production('IdentifierList', 'IdentifierList+'): {
      'identifier_list': 'identifier_list.1'},
    Production('IdentifierList+', 'identifier'): {
      'identifier_list': lambda identifier: [identifier(1)]},
    Production('IdentifierList+', 'identifier', ',', 'IdentifierList+'): {
      'identifier_list': lambda identifier, identifier_list: [identifier(1)] + identifier_list(3)},

    # for function decls
    Production('TypedIdentifierList'): {
      'identifier_list': [], 'type_list': [], 'annotation_list': [], 'mutates_list': []},
    Production('TypedIdentifierList', 'TypedIdentifierList+'): {
      'identifier_list': 'identifier_list.1', 'type_list': 'type_list.1', 'annotation_list': 'annotation_list.1',
      'mutates_list': 'mutates_list.1'},
    Production('TypedIdentifierList+', 'AnnotationList', 'Mutates?', 'identifier', ':', 'Type', 'OptDefaultInit'): {
      'identifier_list': lambda identifier: [identifier(3)],
      'type_list': lambda ast: [ast(5)],
      'annotation_list': lambda annotation_list: [annotation_list(1)],
      'mutates_list': lambda mutates: [mutates(2)]},
    Production('TypedIdentifierList+', 'AnnotationList', 'Mutates?', 'identifier', ':', 'Type', 'OptDefaultInit', ',', 'TypedIdentifierList+'): {  # noqa
      'identifier_list': lambda identifier, identifier_list: [identifier(3)] + identifier_list(8),
      'type_list': lambda ast, type_list: [ast(5)] + type_list(8),
      'annotation_list': lambda annotation_list: [annotation_list(1)] + annotation_list(8),
      'mutates_list': lambda mutates, mutates_list: [mutates(2)] + mutates_list(8)},
    Production('Mutates?'): {'mutates': False},
    Production('Mutates?', 'mutates'): {'mutates': True},

    # for structs
    Production('MemberList'): {
      'identifier_list': [], 'type_list': [], 'annotation_list': []},
    Production('MemberList', 'MemberList+'): {
      'identifier_list': 'identifier_list.1', 'type_list': 'type_list.1', 'annotation_list': 'annotation_list.1'},
    Production('MemberList+', 'AnnotationList', 'identifier', ':', 'Type', 'OptDefaultInit', 'separator'): {
      'identifier_list': lambda identifier: [identifier(2)],
      'type_list': lambda ast: [ast(4)],
      'annotation_list': lambda annotation_list: [annotation_list(1)]},
    Production('MemberList+', 'AnnotationList', 'identifier', ':', 'Type', 'OptDefaultInit', 'separator',
               'MemberList+'): {
      'identifier_list': lambda identifier, identifier_list: [identifier(2)] + identifier_list(7),
      'type_list': lambda ast, type_list: [ast(4)] + type_list(7),
      'annotation_list': lambda annotation_list: [annotation_list(1)] + annotation_list(7)},

    Production('OptDefaultInit'): {},
    Production('OptDefaultInit', '=', 'Expr'): {},
    Production('TemplateIdentifierList'): {
      'identifier_list': []},
    Production('TemplateIdentifierList', '[', 'TemplateIdentifierList+', ']'): {
      'identifier_list': 'identifier_list.2'},
    Production('TemplateIdentifierList+', 'identifier'): {
      'identifier_list': lambda identifier: [identifier(1)]},
    Production('TemplateIdentifierList+', 'identifier', ',', 'TemplateIdentifierList+'): {
      'identifier_list': lambda identifier, identifier_list: [identifier(1)] + identifier_list(3)},
    Production('ExprList'): {
      'val_list': []},
    Production('ExprList', 'ExprList+'): {
      'val_list': 'val_list.1'},
    Production('ExprList+', 'Expr'): {
      'val_list': lambda ast: [ast(1)]},
    Production('ExprList+', 'Expr', ',', 'ExprList+'): {
      'val_list': lambda ast, val_list: [ast(1)] + val_list(3)},
    Production('Type', 'Type', '|', 'IdentifierType'): {
      'ast': lambda _pos, ast: UnionTypeAst(_pos, [ast(1), ast(3)])},
    Production('Type', 'IdentifierType'): {
      'ast': 'ast.1'},
    Production('IdentifierType', 'identifier', 'TemplateTypeList'): {
      'ast': lambda _pos, identifier, type_list: IdentifierTypeAst(
        _pos, type_identifier=identifier(1), templ_types=type_list(2))},
    Production('TemplateTypeList'): {
      'type_list': []},
    Production('TemplateTypeList', '[', 'TemplateTypeList+', ']'): {
      'type_list': 'type_list.2'},
    Production('TemplateTypeList+', 'Type'): {
      'type_list': lambda ast: [ast(1)]},
    Production('TemplateTypeList+', 'Type', ',', 'TemplateTypeList+'): {
      'type_list': lambda ast, type_list: [ast(1)] + type_list(3)},
    Production('IdentifierType', '(', 'Type', ')'): {
      'ast': 'ast.2'},
    Production('ReturnType'): {
      'ast': None, 'annotation_list': None},
    Production('ReturnType', '->', 'AnnotationList', 'Type'): {
      'ast': 'ast.3', 'annotation_list': 'annotation_list.2'},
    Production('Op', 'cmp_op'): {
      'op': 'op.1'},
    Production('Op', 'sum_op'): {
      'op': 'op.1'},
    Production('Op', 'prod_op'): {
      'op': 'op.1'},
    Production('Op', '='): {
      'op': 'op.1'},
  },
  syn_attrs={
    'ast', 'asts', 'stmt_list', 'identifier_list', 'type_list', 'val_list', 'identifier', 'annotation_list', 'mutates_list',
    'mutates', 'op', 'number', 'string'},
  terminal_attr_rules={
    'cmp_op': {'op': lambda value: value},
    '=': {'op': lambda value: value},
    'sum_op': {'op': lambda value: value},
    'prod_op': {'op': lambda value: value},
    'assign_op': {'op': parse_assign_op},
    'identifier': {'identifier': lambda value: value},
    'int': {'number': lambda value: int(value)},
    'long': {'number': lambda value: parse_long(value)},
    'double': {'number': lambda value: parse_double(value)},
    'float': {'number': lambda value: parse_float(value)},
    'char': {'number': lambda value: ord(parse_char(value))},
    'str': {'string': lambda value: parse_string(value)},
    'hex_int': {'number': lambda value: parse_hex_int(value)}
  }
)











SLEEPY_GRAMMAR = SLEEPY_ATTR_GRAMMAR.grammar
SLEEPY_PARSER = ParserGenerator(SLEEPY_GRAMMAR)
