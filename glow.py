#!/usr/bin/env python3
"""
GLOW - A kid-friendly programming language
Version 2.1 - Critical fixes and Phase 2 features
"""

import sys
import os
import time
import random
import math
import json
import traceback
import argparse
import inspect

# Fix for Python 3.14 compatibility with pyreadline
if sys.version_info >= (3, 10):
    import collections.abc
    collections.Callable = collections.abc.Callable

try:
    import readline
    import atexit
    HAVE_READLINE = True
except Exception as e:
    # Create dummy readline if import fails
    class DummyReadline:
        def read_history_file(self, *args, **kwargs): pass
        def write_history_file(self, *args, **kwargs): pass
        def set_history_length(self, *args, **kwargs): pass
        def add_history(self, *args, **kwargs): pass
    
    readline = DummyReadline()
    
    # Dummy atexit functions
    def dummy_register(*args, **kwargs): pass
    atexit.register = dummy_register
    HAVE_READLINE = False

import urllib.request
import urllib.parse
from typing import List, Dict, Any, Optional, Tuple, Union, Callable, Set
from enum import Enum
from pathlib import Path
import hashlib

# ============================================================================
# 1. ERROR HANDLING (IMPROVED)
# ============================================================================

class GlowError(Exception):
    """Base class for GLOW errors"""
    def __init__(self, message, line=None, col=None, source=None):
        self.message = message
        self.line = line
        self.col = col
        self.source = source
        super().__init__(self._format_message())
    
    def _format_message(self):
        emojis = ["🌟", "✨", "⚡", "🌈", "🎯", "⚠️", "❌", "💡"]
        emoji = emojis[hash(self.__class__.__name__) % len(emojis)]
        
        if self.line is not None:
            location = f"at line {self.line}"
            if self.col is not None:
                location += f", column {self.col}"
            return f"{emoji} Oops! {self.message} ({location})"
        return f"{emoji} Oops! {self.message}"
    
    def show_with_context(self):
        """Show error with source context"""
        if self.line is not None and self.source:
            lines = self.source.split('\n')
            if 0 <= self.line - 1 < len(lines):
                context = lines[self.line - 1]
                pointer = ' ' * (self.col - 1 if self.col else 0) + '^'
                return f"{str(self)}\n\n{context}\n{pointer}"
        return str(self)


class LexerError(GlowError):
    """Lexer-specific errors"""
    pass


class ParserError(GlowError):
    """Parser-specific errors"""
    pass


class RuntimeError(GlowError):
    """Runtime execution errors"""
    pass


class TypeError(GlowError):
    """Type-related errors"""
    pass


class ImportError(GlowError):
    """Import-related errors"""
    pass


# ============================================================================
# 2. TOKEN DEFINITIONS (UPDATED)
# ============================================================================

class TokenType(Enum):
    # Single-character tokens
    LEFT_PAREN = "("
    RIGHT_PAREN = ")"
    LEFT_BRACKET = "["
    RIGHT_BRACKET = "]"
    LEFT_BRACE = "{"
    RIGHT_BRACE = "}"
    COMMA = ","
    DOT = "."
    COLON = ":"
    MINUS = "-"
    PLUS = "+"
    STAR = "*"
    SLASH = "/"
    PERCENT = "%"
    BANG = "!"
    EQUAL = "="
    GREATER = ">"
    LESS = "<"
    
    # Multi-character tokens
    BANG_EQUAL = "!="
    EQUAL_EQUAL = "=="
    GREATER_EQUAL = ">="
    LESS_EQUAL = "<="
    ARROW = "->"
    
    
    # Literals
    IDENTIFIER = "IDENTIFIER"
    STRING = "STRING"
    NUMBER = "NUMBER"
    
    # Keywords
    AND = "and"
    AS = "as"
    ASK = "ask"
    DO = "do"
    ELSE = "else"
    END = "end"
    FALSE = "false"
    FOREVER = "forever"
    FOR = "for"
    EACH = "each"
    GLOW = "glow"
    SAY = "say"
    IF = "if"
    IMPORT = "import"
    IN = "in"
    MODULE = "module"
    NOT = "not"
    NULL = "null"
    OR = "or"
    REPEAT = "repeat"
    RETURN = "return"
    SAVE = "save"
    SET = "set"
    TIMES = "times"
    TO = "to"
    TRUE = "true"
    ACTION = "action"
    FROM = "from"
    STOP = "stop"
    CONTINUE = "continue"
    WAIT = "wait"
    WHAT_IS = "what_is"
    WHILE = "while"
    TRY = "try"
    RESCUE = "rescue"
    MAP = "map"  # NEW: for list comprehensions
    WHERE = "where"  # NEW: for filtering
    
    # Multi-word operators (GLOW special)
    IS = "is"
    IS_NOT = "is not"
    IS_BIGGER_THAN = "is bigger than"
    IS_SMALLER_THAN = "is smaller than"
    IS_BIGGER_OR_EQUAL = "is bigger or equal to"
    IS_SMALLER_OR_EQUAL = "is smaller or equal to"
    
    EOF = "EOF"


class Token:
    """Represents a token in the source code"""
    def __init__(self, type: TokenType, lexeme: str, literal: object, 
                 line: int, col: int):
        self.type = type
        self.lexeme = lexeme
        self.literal = literal
        self.line = line
        self.col = col
    
    def __repr__(self):
        return f"Token({self.type}, '{self.lexeme}', {self.literal})"
    
    def __str__(self):
        return f"{self.type.name}:{self.lexeme}"


# ============================================================================
# 3. LEXER (FIXED)
# ============================================================================

class Lexer:
    """Lexer/tokenizer for GLOW source code"""
    
    MULTI_WORD_OPS = [
        ("is bigger or equal to", TokenType.IS_BIGGER_OR_EQUAL),
        ("is smaller or equal to", TokenType.IS_SMALLER_OR_EQUAL),
        ("is bigger than", TokenType.IS_BIGGER_THAN),
        ("is smaller than", TokenType.IS_SMALLER_THAN),
        ("is not", TokenType.IS_NOT),
        ("is", TokenType.IS),
    ]
    
    KEYWORDS = {
        "and": TokenType.AND,
        "as": TokenType.AS,
        "ask": TokenType.ASK,
        "do": TokenType.DO,
        "else": TokenType.ELSE,
        "end": TokenType.END,
        "false": TokenType.FALSE,
        "forever": TokenType.FOREVER,
        "for": TokenType.FOR,
        "each": TokenType.EACH,
        "glow": TokenType.GLOW,
        "say": TokenType.SAY,
        "if": TokenType.IF,
        "import": TokenType.IMPORT,
        "in": TokenType.IN,
        "module": TokenType.MODULE,
        "not": TokenType.NOT,
        "null": TokenType.NULL,
        "or": TokenType.OR,
        "repeat": TokenType.REPEAT,
        "return": TokenType.RETURN,
        "save": TokenType.SAVE,
        "set": TokenType.SET,
        "times": TokenType.TIMES,
        "to": TokenType.TO,
        "true": TokenType.TRUE,
        "action": TokenType.ACTION,
        "from": TokenType.FROM,
        "stop": TokenType.STOP,
        "continue": TokenType.CONTINUE,
        "wait": TokenType.WAIT,
        "what_is": TokenType.WHAT_IS,
        "while": TokenType.WHILE,
        "try": TokenType.TRY,
        "rescue": TokenType.RESCUE,
        "map": TokenType.MAP,    # NEW
        "where": TokenType.WHERE, # NEW
    }
    
    def __init__(self, source: str, filename: str = "<input>"):
        self.source = source
        self.filename = filename
        self.tokens: List[Token] = []
        self.start = 0
        self.current = 0
        self.line = 1
        self.col = 1
        self.errors: List[LexerError] = []
    
    def scan_tokens(self) -> List[Token]:
        """Scan the entire source and return tokens"""
        while not self.is_at_end():
            self.start = self.current
            self.scan_token()
        
        self.add_token(TokenType.EOF, "")
        return self.tokens
    
    def scan_token(self):
        char = self.advance()
        
        if char == '(': self.add_token(TokenType.LEFT_PAREN)
        elif char == ')': self.add_token(TokenType.RIGHT_PAREN)
        elif char == '[': self.add_token(TokenType.LEFT_BRACKET)
        elif char == ']': self.add_token(TokenType.RIGHT_BRACKET)
        elif char == '{': self.add_token(TokenType.LEFT_BRACE)
        elif char == '}': self.add_token(TokenType.RIGHT_BRACE)
        elif char == ',': self.add_token(TokenType.COMMA)
        elif char == '.': self.add_token(TokenType.DOT)
        elif char == ':': self.add_token(TokenType.COLON)
        elif char == '-':
            if self.match('>'):
                self.add_token(TokenType.ARROW)
            else:
                self.add_token(TokenType.MINUS)
        elif char == '+': self.add_token(TokenType.PLUS)
        elif char == '*': self.add_token(TokenType.STAR)
        elif char == '/':
            if self.match('/'):
                while self.peek() != '\n' and not self.is_at_end():
                    self.advance()
            elif self.match('*'):
                self.block_comment()
            else:
                self.add_token(TokenType.SLASH)
        elif char == '%': self.add_token(TokenType.PERCENT)
        elif char == '=': 
            self.add_token(TokenType.EQUAL_EQUAL if self.match('=') else TokenType.EQUAL)
        elif char == '!': 
            self.add_token(TokenType.BANG_EQUAL if self.match('=') else TokenType.BANG)
        elif char == '<': 
            self.add_token(TokenType.LESS_EQUAL if self.match('=') else TokenType.LESS)
        elif char == '>': 
            self.add_token(TokenType.GREATER_EQUAL if self.match('=') else TokenType.GREATER)
        elif char in ' \t\r':
            if char == '\t': self.col += 3
        elif char == '\n':
            self.line += 1
            self.col = 1
        elif char == '"':
            # Check for triple quotes
            if self.peek() == '"' and self.peek_next() == '"':
                self.advance()  # Second "
                self.advance()  # Third "
                self.multiline_string()
            else:
                self.string()
        elif char.isdigit(): self.number()
        elif char.isalpha() or char == '_': self.identifier()
        elif char == '#':
            while self.peek() != '\n' and not self.is_at_end():
                self.advance()
        else: self.error(f"Unexpected character '{char}'")
    
    def multiline_string(self):
        '''Handle triple-quoted strings: """string here""" '''
        start_line = self.line
        start_col = self.col - 3
        
        content = []
        while not self.is_at_end():
            if self.peek() == '"' and self.peek_next() == '"' and self.peek_next_next() == '"':
                self.advance()  # First "
                self.advance()  # Second "
                self.advance()  # Third "
                value = ''.join(content)
                self.add_token(TokenType.STRING, value)
                return
            
            if self.peek() == '\n':
                self.line += 1
                self.col = 1
                content.append('\n')
            elif self.peek() == '\\':
                self.advance()
                if self.peek() in '\\"nt':
                    escape_char = self.peek()
                    self.advance()
                    if escape_char == 'n':
                        content.append('\n')
                    elif escape_char == 't':
                        content.append('\t')
                    elif escape_char == '\\':
                        content.append('\\')
                    elif escape_char == '"':
                        content.append('"')
                else:
                    content.append('\\')
                    content.append(self.peek())
                    self.advance()
            else:
                content.append(self.peek())
            
            self.advance()
        
        self.error("Unterminated multiline string", line=start_line, col=start_col)
    
    def string(self):
        start_line = self.line
        start_col = self.col - 1
        
        content = []
        while self.peek() != '"' and not self.is_at_end():
            if self.peek() == '\n':
                self.line += 1
                self.col = 1
                content.append('\n')
            elif self.peek() == '\\':
                self.advance()
                if self.peek() in '\\"nt{}':
                    escape_char = self.peek()
                    self.advance()
                    if escape_char == 'n':
                        content.append('\n')
                    elif escape_char == 't':
                        content.append('\t')
                    elif escape_char == '\\':
                        content.append('\\')
                    elif escape_char == '"':
                        content.append('"')
                    elif escape_char == '{':
                        content.append('{')  # String interpolation
                    elif escape_char == '}':
                        content.append('}')
                else:
                    content.append('\\')
                    content.append(self.peek())
                    self.advance()
            elif self.peek() == '{':
                # String interpolation start
                content.append('{')
                self.advance()
            elif self.peek() == '}':
                # String interpolation end
                content.append('}')
                self.advance()
            else:
                content.append(self.peek())
                self.advance()
        
        if self.is_at_end():
            self.error("Unterminated string", line=start_line, col=start_col)
            return
        
        self.advance()  # Closing quote
        value = ''.join(content)
        self.add_token(TokenType.STRING, value)
    
    def block_comment(self):
        """Handle /* ... */ comments"""
        while not self.is_at_end():
            if self.peek() == '*' and self.peek_next() == '/':
                self.advance()  # *
                self.advance()  # /
                return
            if self.peek() == '\n':
                self.line += 1
                self.col = 1
            self.advance()
        
        self.error("Unterminated block comment")
    
    def identifier(self):
        while self.peek().isalnum() or self.peek() == '_':
            self.advance()
        
        text = self.source[self.start:self.current]
        
        # Check for multi-word operators
        for op_text, op_type in self.MULTI_WORD_OPS:
            if text.lower() == op_text.split()[0]:
                saved_current = self.current
                saved_line = self.line
                saved_col = self.col
                
                words = [text]
                while len(words) < len(op_text.split()):
                    while self.peek() in ' \t':
                        self.advance()
                    
                    start = self.current
                    while self.peek().isalpha():
                        self.advance()
                    next_word = self.source[start:self.current]
                    
                    if next_word:
                        words.append(next_word)
                    else:
                        break
                
                full_text = ' '.join(words)
                if full_text.lower() == op_text:
                    self.add_token(op_type, full_text)
                    return
                else:
                    self.current = saved_current
                    self.line = saved_line
                    self.col = saved_col
                    text = self.source[self.start:self.current]
                    break
        
        # Check for 'in' operator
        if text == 'in':
            self.add_token(TokenType.IN)
            return
        
        token_type = self.KEYWORDS.get(text.lower())
        if token_type:
            self.add_token(token_type, text)
        else:
            self.add_token(TokenType.IDENTIFIER, text)
    
    def number(self):
        while self.peek().isdigit():
            self.advance()
        
        if self.peek() == '.' and self.peek_next().isdigit():
            self.advance()
            while self.peek().isdigit():
                self.advance()
        
        value = float(self.source[self.start:self.current])
        self.add_token(TokenType.NUMBER, value)
    
    def add_token(self, type: TokenType, literal: object = None):
        text = self.source[self.start:self.current]
        self.tokens.append(Token(type, text, literal, self.line, self.col))
    
    def advance(self):
        char = self.source[self.current]
        self.current += 1
        self.col += 1
        return char
    
    def match(self, expected: str) -> bool:
        if self.is_at_end():
            return False
        if self.source[self.current] != expected:
            return False
        self.current += 1
        self.col += 1
        return True
    
    def peek(self) -> str:
        if self.is_at_end():
            return '\0'
        return self.source[self.current]
    
    def peek_next(self) -> str:
        if self.current + 1 >= len(self.source):
            return '\0'
        return self.source[self.current + 1]
    
    def peek_next_next(self) -> str:
        if self.current + 2 >= len(self.source):
            return '\0'
        return self.source[self.current + 2]
    
    def is_at_end(self) -> bool:
        return self.current >= len(self.source)
    
    def error(self, message: str, line: int = None, col: int = None):
        if line is None: line = self.line
        if col is None: col = self.col
        self.errors.append(LexerError(message, line, col, self.source))
    
    def has_errors(self) -> bool:
        return len(self.errors) > 0


# ============================================================================
# 4. AST NODES (ENHANCED)
# ============================================================================

class ASTNode:
    """Base class for all AST nodes"""
    def __init__(self, line: int = 0, col: int = 0):
        self.line = line
        self.col = col
    
    def accept(self, visitor):
        raise NotImplementedError


class Program(ASTNode):
    def __init__(self, statements: List[ASTNode]):
        super().__init__()
        self.statements = statements
    
    def __repr__(self):
        return f"Program({len(self.statements)} statements)"
    
    def accept(self, visitor):
        return visitor.visit_program(self)


class Literal(ASTNode):
    def __init__(self, value, line=0, col=0):
        super().__init__(line, col)
        self.value = value
    
    def __repr__(self):
        return f"Literal({self.value!r})"
    
    def accept(self, visitor):
        return visitor.visit_literal(self)


class InterpolatedString(ASTNode):  # NEW
    """String with interpolated expressions: "Hello {name}" """
    def __init__(self, parts: List[Union[str, ASTNode]], line=0, col=0):
        super().__init__(line, col)
        self.parts = parts  # Alternating string literals and expressions
    
    def __repr__(self):
        return f"InterpolatedString({len(self.parts)} parts)"
    
    def accept(self, visitor):
        return visitor.visit_interpolated_string(self)


class Identifier(ASTNode):
    def __init__(self, name: str, line=0, col=0):
        super().__init__(line, col)
        self.name = name
    
    def __repr__(self):
        return f"Identifier({self.name!r})"
    
    def accept(self, visitor):
        return visitor.visit_identifier(self)


class BinaryOp(ASTNode):
    def __init__(self, left: ASTNode, operator: str, right: ASTNode, line=0, col=0):
        super().__init__(line, col)
        self.left = left
        self.operator = operator
        self.right = right
    
    def __repr__(self):
        return f"BinaryOp({self.left} {self.operator} {self.right})"
    
    def accept(self, visitor):
        return visitor.visit_binary_op(self)


class UnaryOp(ASTNode):
    def __init__(self, operator: str, right: ASTNode, line=0, col=0):
        super().__init__(line, col)
        self.operator = operator
        self.right = right
    
    def __repr__(self):
        return f"UnaryOp({self.operator} {self.right})"
    
    def accept(self, visitor):
        return visitor.visit_unary_op(self)


class SetStatement(ASTNode):
    def __init__(self, name: Identifier, value: ASTNode, line=0, col=0):
        super().__init__(line, col)
        self.name = name
        self.value = value
    
    def __repr__(self):
        return f"SetStatement({self.name.name} = {self.value})"
    
    def accept(self, visitor):
        return visitor.visit_set_statement(self)


class SetPropertyStatement(ASTNode):  # NEW: For obj.property = value
    def __init__(self, object_expr: ASTNode, property_name: str, value: ASTNode, line=0, col=0):
        super().__init__(line, col)
        self.object_expr = object_expr
        self.property_name = property_name
        self.value = value
    
    def __repr__(self):
        return f"SetPropertyStatement({self.object_expr}.{self.property_name} = {self.value})"
    
    def accept(self, visitor):
        return visitor.visit_set_property_statement(self)


class GlowStatement(ASTNode):
    def __init__(self, expression: ASTNode, line=0, col=0):
        super().__init__(line, col)
        self.expression = expression
    
    def __repr__(self):
        return f"GlowStatement({self.expression})"
    
    def accept(self, visitor):
        return visitor.visit_glow_statement(self)


class AskStatement(ASTNode):
    def __init__(self, prompt: str, name: Identifier, line=0, col=0):
        super().__init__(line, col)
        self.prompt = prompt
        self.name = name
    
    def __repr__(self):
        return f"AskStatement(prompt={self.prompt!r}, name={self.name.name})"
    
    def accept(self, visitor):
        return visitor.visit_ask_statement(self)


class IfStatement(ASTNode):
    def __init__(self, condition: ASTNode, then_branch: List[ASTNode], 
                 else_branch: Optional[List[ASTNode]] = None, line=0, col=0):
        super().__init__(line, col)
        self.condition = condition
        self.then_branch = then_branch
        self.else_branch = else_branch or []
    
    def __repr__(self):
        return f"IfStatement(if {self.condition} then {len(self.then_branch)} statements)"
    
    def accept(self, visitor):
        return visitor.visit_if_statement(self)


class RepeatStatement(ASTNode):
    def __init__(self, count: Optional[ASTNode], body: List[ASTNode], 
                 is_forever: bool = False, line=0, col=0):
        super().__init__(line, col)
        self.count = count
        self.body = body
        self.is_forever = is_forever
    
    def __repr__(self):
        if self.is_forever:
            return f"RepeatStatement(forever, {len(self.body)} statements)"
        return f"RepeatStatement({self.count} times, {len(self.body)} statements)"
    
    def accept(self, visitor):
        return visitor.visit_repeat_statement(self)


class ForEachStatement(ASTNode):
    def __init__(self, variable: Identifier, iterable: ASTNode, 
                 body: List[ASTNode], index_var: Optional[Identifier] = None, line=0, col=0):
        super().__init__(line, col)
        self.variable = variable
        self.iterable = iterable
        self.body = body
        self.index_var = index_var
    
    def __repr__(self):
        if self.index_var:
            return f"ForEachStatement({self.variable.name}, {self.index_var.name} in {self.iterable})"
        return f"ForEachStatement({self.variable.name} in {self.iterable})"
    
    def accept(self, visitor):
        return visitor.visit_for_each_statement(self)


class WhileStatement(ASTNode):
    def __init__(self, condition: ASTNode, body: List[ASTNode], line=0, col=0):
        super().__init__(line, col)
        self.condition = condition
        self.body = body
    
    def __repr__(self):
        return f"WhileStatement(while {self.condition} do {len(self.body)} statements)"
    
    def accept(self, visitor):
        return visitor.visit_while_statement(self)


class TryStatement(ASTNode):
    def __init__(self, try_body: List[ASTNode], rescue_body: List[ASTNode], 
                 error_var: Optional[Identifier] = None, line=0, col=0):
        super().__init__(line, col)
        self.try_body = try_body
        self.rescue_body = rescue_body
        self.error_var = error_var
    
    def __repr__(self):
        return f"TryStatement(try {len(self.try_body)} statements, rescue {len(self.rescue_body)} statements)"
    
    def accept(self, visitor):
        return visitor.visit_try_statement(self)


class ActionStatement(ASTNode):
    def __init__(self, name: Identifier, params: List[Identifier], 
                 body: List[ASTNode], line=0, col=0):
        super().__init__(line, col)
        self.name = name
        self.params = params
        self.body = body
    
    def __repr__(self):
        return f"ActionStatement({self.name.name}({', '.join(p.name for p in self.params)}))"
    
    def accept(self, visitor):
        return visitor.visit_action_statement(self)


class CallExpression(ASTNode):
    def __init__(self, callee: ASTNode, args: List[ASTNode], line=0, col=0):
        super().__init__(line, col)
        self.callee = callee
        self.args = args
    
    def __repr__(self):
        return f"CallExpression({self.callee}({len(self.args)} args))"
    
    def accept(self, visitor):
        return visitor.visit_call_expression(self)


class MethodCallExpression(ASTNode):  # NEW: For obj.method(arg1, arg2)
    def __init__(self, object_expr: ASTNode, method_name: str, args: List[ASTNode], line=0, col=0):
        super().__init__(line, col)
        self.object_expr = object_expr
        self.method_name = method_name
        self.args = args
    
    def __repr__(self):
        return f"MethodCallExpression({self.object_expr}.{self.method_name}({len(self.args)} args))"
    
    def accept(self, visitor):
        return visitor.visit_method_call_expression(self)


class DoCallExpression(ASTNode):
    """Special AST node for do calls without parentheses"""
    def __init__(self, callee: Identifier, args: List[ASTNode], line=0, col=0):
        super().__init__(line, col)
        self.callee = callee
        self.args = args
    
    def __repr__(self):
        return f"DoCallExpression({self.callee.name} {len(self.args)} args)"
    
    def accept(self, visitor):
        return visitor.visit_do_call_expression(self)


class ReturnStatement(ASTNode):
    def __init__(self, value: Optional[ASTNode] = None, line=0, col=0):
        super().__init__(line, col)
        self.value = value
    
    def __repr__(self):
        return f"ReturnStatement({self.value})"
    
    def accept(self, visitor):
        return visitor.visit_return_statement(self)


class ListLiteral(ASTNode):
    def __init__(self, elements: List[ASTNode], line=0, col=0):
        super().__init__(line, col)
        self.elements = elements
    
    def __repr__(self):
        return f"ListLiteral([{', '.join(str(e) for e in self.elements)}])"
    
    def accept(self, visitor):
        return visitor.visit_list_literal(self)


class ObjectLiteral(ASTNode):
    def __init__(self, properties: Dict[str, ASTNode], line=0, col=0):
        super().__init__(line, col)
        self.properties = properties
    
    def __repr__(self):
        props = ', '.join(f'{k}: {v}' for k, v in self.properties.items())
        return f"ObjectLiteral({{{props}}})"
    
    def accept(self, visitor):
        return visitor.visit_object_literal(self)


class GetProperty(ASTNode):
    def __init__(self, object_expr: ASTNode, name: str, line=0, col=0):
        super().__init__(line, col)
        self.object_expr = object_expr
        self.name = name
    
    def __repr__(self):
        return f"GetProperty({self.object_expr}.{self.name})"
    
    def accept(self, visitor):
        return visitor.visit_get_property(self)


class SetProperty(ASTNode):
    def __init__(self, object_expr: ASTNode, index: ASTNode, value: ASTNode, line=0, col=0):
        super().__init__(line, col)
        self.object_expr = object_expr
        self.index = index
        self.value = value
    
    def __repr__(self):
        return f"SetProperty({self.object_expr}[{self.index}] = {self.value})"
    
    def accept(self, visitor):
        return visitor.visit_set_property(self)


class IndexExpression(ASTNode):
    def __init__(self, object_expr: ASTNode, index: ASTNode, line=0, col=0):
        super().__init__(line, col)
        self.object_expr = object_expr
        self.index = index
    
    def __repr__(self):
        return f"IndexExpression({self.object_expr}[{self.index}])"
    
    def accept(self, visitor):
        return visitor.visit_index_expression(self)


class SliceExpression(ASTNode):  # NEW: For array slicing [start:end]
    def __init__(self, object_expr: ASTNode, start: Optional[ASTNode], 
                 end: Optional[ASTNode], step: Optional[ASTNode] = None, line=0, col=0):
        super().__init__(line, col)
        self.object_expr = object_expr
        self.start = start
        self.end = end
        self.step = step
    
    def __repr__(self):
        return f"SliceExpression({self.object_expr}[{self.start}:{self.end}:{self.step}])"
    
    def accept(self, visitor):
        return visitor.visit_slice_expression(self)


class InExpression(ASTNode):  # NEW: For "item in list"
    def __init__(self, item: ASTNode, container: ASTNode, line=0, col=0):
        super().__init__(line, col)
        self.item = item
        self.container = container
    
    def __repr__(self):
        return f"InExpression({self.item} in {self.container})"
    
    def accept(self, visitor):
        return visitor.visit_in_expression(self)


class MapExpression(ASTNode):  # NEW: For list comprehensions
    def __init__(self, list_expr: ASTNode, variable: Identifier, 
                 transform: ASTNode, filter_cond: Optional[ASTNode] = None, line=0, col=0):
        super().__init__(line, col)
        self.list_expr = list_expr
        self.variable = variable
        self.transform = transform  # Expression to apply to each element
        self.filter_cond = filter_cond  # Optional filter condition
    
    def __repr__(self):
        return f"MapExpression(map {self.list_expr} with {self.variable.name})"
    
    def accept(self, visitor):
        return visitor.visit_map_expression(self)


class StopStatement(ASTNode):
    def __init__(self, line=0, col=0):
        super().__init__(line, col)
    
    def __repr__(self):
        return f"StopStatement()"
    
    def accept(self, visitor):
        return visitor.visit_stop_statement(self)


class ContinueStatement(ASTNode):
    def __init__(self, line=0, col=0):
        super().__init__(line, col)
    
    def __repr__(self):
        return f"ContinueStatement()"
    
    def accept(self, visitor):
        return visitor.visit_continue_statement(self)


class WaitStatement(ASTNode):
    def __init__(self, seconds: ASTNode, line=0, col=0):
        super().__init__(line, col)
        self.seconds = seconds
    
    def __repr__(self):
        return f"WaitStatement({self.seconds})"
    
    def accept(self, visitor):
        return visitor.visit_wait_statement(self)


class ImportStatement(ASTNode):
    def __init__(self, path: str, alias: Optional[str] = None, 
                 imports: Optional[List[str]] = None, line=0, col=0):
        super().__init__(line, col)
        self.path = path
        self.alias = alias
        self.imports = imports
    
    def __repr__(self):
        if self.imports:
            return f"ImportStatement(from {self.path} import {', '.join(self.imports)})"
        elif self.alias:
            return f"ImportStatement(import {self.path} as {self.alias})"
        else:
            return f"ImportStatement(import {self.path})"
    
    def accept(self, visitor):
        return visitor.visit_import_statement(self)


# ============================================================================
# 5. PARSER (COMPLETELY FIXED)
# ============================================================================

class Parser:
    """Recursive descent parser for GLOW - FIXED VERSION"""
    
    def __init__(self, tokens, source: str = "<input>"):
        self.tokens = tokens
        self.source = source
        self.current = 0
        self.errors: List[ParserError] = []
    
    def parse(self) -> Program:
        statements = []
        
        while not self.is_at_end():
            try:
                stmt = self.statement()
                if stmt:
                    statements.append(stmt)
            except ParserError as e:
                self.errors.append(e)
                self.synchronize()
        
        return Program(statements)
    
    def statement(self) -> Optional[ASTNode]:
        if self.check("GLOW") or self.check("SAY"):
            line, col = self.current_location()
            token_type = self.peek().type
            self.advance()
            return self.glow_statement(line, col, token_type == TokenType.SAY)
        elif self.check("SET"):
            self.advance()
            return self.set_statement()
        elif self.check("ASK"):
            self.advance()
            return self.ask_statement()
        elif self.check("IF"):
            self.advance()
            return self.if_statement()
        elif self.check("REPEAT"):
            self.advance()
            return self.repeat_statement()
        elif self.check("FOR"):
            self.advance()
            return self.for_each_statement()
        elif self.check("WHILE"):
            self.advance()
            return self.while_statement()
        elif self.check("TRY"):
            self.advance()
            return self.try_statement()
        elif self.check("ACTION"):
            self.advance()
            return self.action_statement()
        elif self.check("RETURN"):
            self.advance()
            return self.return_statement()
        elif self.check("DO"):
            return self.do_call_statement()
        elif self.check("IMPORT"):
            self.advance()
            return self.import_statement()
        elif self.check("STOP"):
            line, col = self.current_location()
            self.advance()
            return StopStatement(line, col)
        elif self.check("CONTINUE"):
            line, col = self.current_location()
            self.advance()
            return ContinueStatement(line, col)
        elif self.check("WAIT"):
            self.advance()
            return self.wait_statement()
        elif self.peek_type() == "IDENTIFIER" and self.peek_next_type() == "LEFT_PAREN":
            return self.expression_statement()
        else:
            expr = self.expression()
            if isinstance(expr, (CallExpression, MethodCallExpression, DoCallExpression)):
                return expr
            return expr
    
    def set_statement(self) -> Union[SetStatement, SetProperty, SetPropertyStatement]:
        line, col = self.previous_location()
        
        # Try to parse as property assignment: obj.property or obj[index]
        expr = self.primary()
        
        # Check for property access
        while True:
            if self.check("DOT"):
                self.advance()
                if not self.check("IDENTIFIER"):
                    self.error("Expected property name after '.'")
                prop_token = self.advance()
                expr = GetProperty(expr, prop_token.lexeme, line, col)
            elif self.check("LEFT_BRACKET"):
                self.advance()
                index = self.expression()
                if not index:
                    self.error("Expected index expression inside '[]'")
                
                # Check for slice syntax
                if self.check("COLON"):
                    # This is a slice, not an index assignment
                    # We'll parse it but assignment to slices isn't supported
                    self.advance()
                    end = self.expression() if not self.check("RIGHT_BRACKET") else None
                    
                    # Check for step
                    step = None
                    if self.check("COLON"):
                        self.advance()
                        step = self.expression() if not self.check("RIGHT_BRACKET") else None
                    
                    if not self.check("RIGHT_BRACKET"):
                        self.error("Expected ']' after slice")
                    self.advance()
                    
                    expr = SliceExpression(expr, index, end, step, line, col)
                else:
                    if not self.check("RIGHT_BRACKET"):
                        self.error("Expected ']' after index")
                    self.advance()
                    expr = IndexExpression(expr, index, line, col)
            else:
                break
        
        # Now we have the left-hand side, check for 'to'
        if not self.check("TO"):
            self.error("Expected 'to' in set statement")
        self.advance()
        
        value = self.expression()
        if not value:
            self.error("Expected expression after 'to'")
        
        # Create appropriate assignment node
        if isinstance(expr, Identifier):
            return SetStatement(expr, value, line, col)
        elif isinstance(expr, GetProperty):
            return SetPropertyStatement(expr.object_expr, expr.name, value, line, col)
        elif isinstance(expr, IndexExpression):
            return SetProperty(expr.object_expr, expr.index, value, line, col)
        else:
            self.error(f"Cannot assign to {expr.__class__.__name__}")
    
    def try_statement(self) -> TryStatement:
        line, col = self.previous_location()
        
        # Parse try block
        try_body = self.block()
        
        # Parse rescue
        if not self.check("RESCUE"):
            self.error("Expected 'rescue' after try block")
        self.advance()
        
        # Optional error variable
        error_var = None
        if self.check("AS"):
            self.advance()
            if not self.check("IDENTIFIER"):
                self.error("Expected error variable name after 'as'")
            error_token = self.advance()
            error_var = Identifier(error_token.lexeme, error_token.line, error_token.col)
        
        rescue_body = self.block()
        
        if not self.check("END"):
            self.error("Expected 'end' after try/rescue block")
        self.advance()
        
        return TryStatement(try_body, rescue_body, error_var, line, col)
    
    def while_statement(self) -> WhileStatement:
        line, col = self.previous_location()
        
        condition = self.expression()
        if not condition:
            self.error("Expected condition after 'while'")
        
        body = self.block()
        
        if not self.check("END"):
            self.error("Expected 'end' after while block")
        self.advance()
        
        return WhileStatement(condition, body, line, col)
    
    def for_each_statement(self) -> ForEachStatement:
        line, col = self.previous_location()
        
        if not self.check("EACH"):
            self.error("Expected 'each' after 'for'")
        self.advance()
        
        if not self.check("IDENTIFIER"):
            self.error("Expected variable name after 'each'")
        
        token = self.advance()
        variable = Identifier(token.lexeme, line, col)
        
        # Check for optional index variable
        index_var = None
        if self.check("COMMA"):
            self.advance()
            if not self.check("IDENTIFIER"):
                self.error("Expected index variable name after comma")
            index_token = self.advance()
            index_var = Identifier(index_token.lexeme, index_token.line, index_token.col)
        
        if not self.check("IN"):
            self.error("Expected 'in' after variable name")
        self.advance()
        
        iterable = self.expression()
        if not iterable:
            self.error("Expected iterable expression after 'in'")
        
        body = self.block()
        
        if not self.check("END"):
            self.error("Expected 'end' after for-each block")
        self.advance()
        
        return ForEachStatement(variable, iterable, body, index_var, line, col)
    
    def import_statement(self) -> ImportStatement:
        line, col = self.previous_location()
        
        # Check for "from X import Y" syntax
        if self.check("STRING"):
            # Simple import "path"
            path_token = self.advance()
            path = path_token.literal
            
            alias = None
            if self.check("AS"):
                self.advance()
                if not self.check("IDENTIFIER"):
                    self.error("Expected module alias after 'as'")
                alias_token = self.advance()
                alias = alias_token.lexeme
            
            return ImportStatement(path, alias, None, line, col)
        elif self.check("IDENTIFIER"):
            # "from module import item1, item2"
            module_token = self.advance()
            module_name = module_token.lexeme
            
            if not self.check("IMPORT"):
                self.error("Expected 'import' after module name")
            self.advance()
            
            imports = []
            if not self.check("IDENTIFIER"):
                self.error("Expected at least one import item")
            
            # Parse import items
            while True:
                if not self.check("IDENTIFIER"):
                    self.error("Expected identifier in import list")
                import_token = self.advance()
                imports.append(import_token.lexeme)
                
                if not self.check("COMMA"):
                    break
                self.advance()
            
            return ImportStatement(module_name, None, imports, line, col)
        else:
            self.error("Expected string path or module name after 'import'")
    
    def do_call_statement(self) -> DoCallExpression:
        line, col = self.current_location()
        self.advance()  # Skip 'do'
        
        if not self.check("IDENTIFIER"):
            self.error("Expected function name after 'do'")
        
        token = self.advance()
        callee = Identifier(token.lexeme, line, col)
        
        # Parse space-separated arguments
        args = []
        while self._can_start_expression():
            args.append(self.expression())
        
        return DoCallExpression(callee, args, line, col)
    
    def do_call_expression(self) -> DoCallExpression:
        line, col = self.previous_location()
        
        if not self.check("IDENTIFIER"):
            self.error("Expected function name after 'do'")
        
        token = self.advance()
        callee = Identifier(token.lexeme, line, col)
        
        # Parse space-separated arguments
        args = []
        while self._can_start_expression():
            args.append(self.expression())
        
        return DoCallExpression(callee, args, line, col)
    
    def _can_start_expression(self) -> bool:
        if self.is_at_end():
            return False
        
        token_type = self.peek().type
        return token_type in [
            TokenType.NUMBER, TokenType.STRING, TokenType.IDENTIFIER,
            TokenType.LEFT_PAREN, TokenType.LEFT_BRACKET, TokenType.LEFT_BRACE,
            TokenType.TRUE, TokenType.FALSE, TokenType.NULL,
            TokenType.MINUS, TokenType.NOT, TokenType.DO
        ]
    
    def glow_statement(self, line: int, col: int, is_say: bool = False) -> GlowStatement:
        value = self.expression()
        if not value:
            self.error("Expected expression after 'glow' or 'say'")
        return GlowStatement(value, line, col)
    
    def wait_statement(self) -> WaitStatement:
        line, col = self.previous_location()
        seconds = self.expression()
        if not seconds:
            self.error("Expected number of seconds after 'wait'")
        return WaitStatement(seconds, line, col)
    
    def ask_statement(self) -> AskStatement:
        line, col = self.previous_location()
        
        if not self.check("STRING"):
            self.error("Expected string prompt after 'ask'")
        
        token = self.advance()
        prompt = token.literal
        
        if not self.check("SAVE"):
            self.error("Expected 'save' after prompt")
        self.advance()
        
        if not self.check("AS"):
            self.error("Expected 'as' after 'save'")
        self.advance()
        
        if not self.check("IDENTIFIER"):
            self.error("Expected variable name after 'as'")
        
        token = self.advance()
        name = Identifier(token.lexeme, line, col)
        
        return AskStatement(prompt, name, line, col)
    
    def if_statement(self) -> IfStatement:
        line, col = self.previous_location()
        
        condition = self.expression()
        if not condition:
            self.error("Expected condition after 'if'")
        
        then_branch = self.block()
        
        else_branch = []
        if self.check("ELSE"):
            self.advance()
            else_branch = self.block()
        
        if not self.check("END"):
            self.error("Expected 'end' after if statement")
        self.advance()
        
        return IfStatement(condition, then_branch, else_branch, line, col)
    
    def repeat_statement(self) -> RepeatStatement:
        line, col = self.previous_location()
        
        if self.check("FOREVER"):
            self.advance()
            body = self.block()
            if not self.check("END"):
                self.error("Expected 'end' after repeat block")
            self.advance()
            return RepeatStatement(None, body, is_forever=True, line=line, col=col)
        
        count = self.expression()
        if not count:
            self.error("Expected count or 'forever' after 'repeat'")
        
        if not self.check("TIMES"):
            self.error("Expected 'times' after repeat count")
        self.advance()
        
        body = self.block()
        
        if not self.check("END"):
            self.error("Expected 'end' after repeat block")
        self.advance()
        
        return RepeatStatement(count, body, is_forever=False, line=line, col=col)
    
    def action_statement(self) -> ActionStatement:
        line, col = self.previous_location()
        
        if not self.check("IDENTIFIER"):
            self.error("Expected action name after 'action'")
        
        token = self.advance()
        name = Identifier(token.lexeme, line, col)
        
        params = []
        while self.check("IDENTIFIER"):
            param_line, param_col = self.current_location()
            token = self.advance()
            param = Identifier(token.lexeme, param_line, param_col)
            params.append(param)
        
        body = self.block()
        
        if not self.check("END"):
            self.error("Expected 'end' after action body")
        self.advance()
        
        return ActionStatement(name, params, body, line, col)
    
    def return_statement(self) -> ReturnStatement:
        line, col = self.previous_location()
        
        if self.peek_type() not in ["END", "EOF", "ELSE"]:
            value = self.expression()
            return ReturnStatement(value, line, col)
        
        return ReturnStatement(None, line, col)
    
    def expression_statement(self) -> Union[CallExpression, MethodCallExpression, DoCallExpression]:
        expr = self.expression()
        if isinstance(expr, (CallExpression, MethodCallExpression, DoCallExpression)):
            return expr
        return expr
    
    def expression(self) -> Optional[ASTNode]:
        return self.assignment()
    
    def assignment(self) -> Optional[ASTNode]:
        expr = self.or_expression()
        return expr
    
    def or_expression(self) -> ASTNode:
        expr = self.and_expression()
        
        while self.check("OR"):
            operator = self.peek().lexeme
            self.advance()
            right = self.and_expression()
            expr = BinaryOp(expr, operator, right, expr.line, expr.col)
        
        return expr
    
    def and_expression(self) -> ASTNode:
        expr = self.equality()
        
        while self.check("AND"):
            operator = self.peek().lexeme
            self.advance()
            right = self.equality()
            expr = BinaryOp(expr, operator, right, expr.line, expr.col)
        
        return expr
    
    def equality(self) -> ASTNode:
        expr = self.comparison()
        
        while self.check("IS") or self.check("IS_NOT"):
            operator = self.peek().lexeme
            self.advance()
            right = self.comparison()
            expr = BinaryOp(expr, operator, right, expr.line, expr.col)
        
        return expr
    
    def comparison(self) -> ASTNode:
        expr = self.addition()
        
        while (self.check("IS_BIGGER_THAN") or self.check("IS_SMALLER_THAN") or
               self.check("IS_BIGGER_OR_EQUAL") or self.check("IS_SMALLER_OR_EQUAL") or
               self.check("LESS") or self.check("GREATER") or 
               self.check("LESS_EQUAL") or self.check("GREATER_EQUAL")):
            operator = self.peek().lexeme
            self.advance()
            right = self.addition()
            expr = BinaryOp(expr, operator, right, expr.line, expr.col)
        
        return expr
    
    def addition(self) -> ASTNode:
        expr = self.multiplication()
        
        while self.check("PLUS") or self.check("MINUS"):
            operator = self.peek().lexeme
            self.advance()
            right = self.multiplication()
            expr = BinaryOp(expr, operator, right, expr.line, expr.col)
        
        return expr
    
    def multiplication(self) -> ASTNode:
        expr = self.unary()
        
        while self.check("STAR") or self.check("SLASH") or self.check("PERCENT"):
            operator = self.peek().lexeme
            self.advance()
            right = self.unary()
            expr = BinaryOp(expr, operator, right, expr.line, expr.col)
        
        return expr
    
    def unary(self) -> ASTNode:
        if self.check("NOT") or self.check("MINUS"):
            operator = self.peek().lexeme
            line, col = self.current_location()
            self.advance()
            right = self.unary()
            return UnaryOp(operator, right, line, col)
        
        return self.in_expression()
    
    def in_expression(self) -> ASTNode:
        expr = self.call()
        
        # Handle 'in' operator
        if self.check("IN"):
            line, col = self.current_location()
            self.advance()
            container = self.call()
            return InExpression(expr, container, line, col)
        
        return expr
    
    def call(self) -> ASTNode:
        expr = self.primary()
        
        while True:
            if self.check("LEFT_PAREN"):
                # Check if this is a method call
                if isinstance(expr, GetProperty):
                    # Convert GetProperty to MethodCallExpression
                    self.advance()
                    args = []
                    if not self.check("RIGHT_PAREN"):
                        args.append(self.expression())
                        while self.check("COMMA"):
                            self.advance()
                            args.append(self.expression())
                    
                    if not self.check("RIGHT_PAREN"):
                        self.error("Expected ')' after arguments")
                    self.advance()
                    
                    expr = MethodCallExpression(expr.object_expr, expr.name, args, expr.line, expr.col)
                else:
                    # Regular function call
                    self.advance()
                    args = []
                    if not self.check("RIGHT_PAREN"):
                        args.append(self.expression())
                        while self.check("COMMA"):
                            self.advance()
                            args.append(self.expression())
                    
                    if not self.check("RIGHT_PAREN"):
                        self.error("Expected ')' after arguments")
                    self.advance()
                    
                    expr = CallExpression(expr, args, expr.line, expr.col)
            elif self.check("DOT"):
                self.advance()
                if not self.check("IDENTIFIER"):
                    self.error("Expected property name after '.'")
                prop_token = self.advance()
                expr = GetProperty(expr, prop_token.lexeme, expr.line, expr.col)
            elif self.check("LEFT_BRACKET"):
                self.advance()
                index = self.expression()
                if not index:
                    self.error("Expected index expression inside '[]'")
                
                # Check for slice syntax
                if self.check("COLON"):
                    # This is a slice
                    self.advance()
                    end = self.expression() if not self.check("RIGHT_BRACKET") else None
                    
                    # Check for step
                    step = None
                    if self.check("COLON"):
                        self.advance()
                        step = self.expression() if not self.check("RIGHT_BRACKET") else None
                    
                    if not self.check("RIGHT_BRACKET"):
                        self.error("Expected ']' after slice")
                    self.advance()
                    
                    expr = SliceExpression(expr, index, end, step, expr.line, expr.col)
                else:
                    if not self.check("RIGHT_BRACKET"):
                        self.error("Expected ']' after index")
                    self.advance()
                    expr = IndexExpression(expr, index, expr.line, expr.col)
            else:
                break
        
        return expr
    
    def primary(self) -> ASTNode:
        if self.check("FALSE"):
            line, col = self.current_location()
            self.advance()
            return Literal(False, line, col)
        elif self.check("TRUE"):
            line, col = self.current_location()
            self.advance()
            return Literal(True, line, col)
        elif self.check("NULL"):
            line, col = self.current_location()
            self.advance()
            return Literal(None, line, col)
        elif self.check("NUMBER"):
            line, col = self.current_location()
            token = self.advance()
            return Literal(token.literal, line, col)
        elif self.check("STRING"):
            line, col = self.current_location()
            token = self.advance()
            # Check for interpolation in string
            text = token.literal
            if '{' in text:
                parts = self._parse_string_interpolation(text, line, col)
                return InterpolatedString(parts, line, col)
            else:
                return Literal(text, line, col)
        elif self.check("IDENTIFIER"):
            line, col = self.current_location()
            token = self.advance()
            return Identifier(token.lexeme, line, col)
        elif self.check("LEFT_PAREN"):
            self.advance()
            expr = self.expression()
            if not self.check("RIGHT_PAREN"):
                self.error("Expected ')' after expression")
            self.advance()
            return expr
        elif self.check("LEFT_BRACKET"):
            line, col = self.current_location()
            self.advance()
            
            # Check for map expression: [x * 2 for x in items]
            if self.check("MAP"):
                self.advance()
                return self.map_expression(line, col)
            
            elements = []
            if not self.check("RIGHT_BRACKET"):
                elements.append(self.expression())
                while self.check("COMMA"):
                    self.advance()
                    elements.append(self.expression())
            
            if not self.check("RIGHT_BRACKET"):
                self.error("Expected ']' after list elements")
            self.advance()
            
            return ListLiteral(elements, line, col)
        elif self.check("LEFT_BRACE"):
            line, col = self.current_location()
            self.advance()
            properties = {}
            
            if not self.check("RIGHT_BRACE"):
                # Parse key-value pairs
                while True:
                    # Key must be identifier or string
                    if self.check("IDENTIFIER"):
                        key_token = self.advance()
                        key = key_token.lexeme
                    elif self.check("STRING"):
                        key_token = self.advance()
                        key = key_token.literal
                    else:
                        self.error("Expected property name or string key")
                    
                    if not self.check("COLON"):
                        self.error("Expected ':' after property name")
                    self.advance()
                    
                    value = self.expression()
                    properties[key] = value
                    
                    if not self.check("COMMA"):
                        break
                    self.advance()
            
            if not self.check("RIGHT_BRACE"):
                self.error("Expected '}' after object properties")
            self.advance()
            
            return ObjectLiteral(properties, line, col)
        elif self.check("DO"):
            # Handle do calls as expressions (e.g., set x to do add 1 2)
            self.advance()
            return self.do_call_expression()
        else:
            self.error("Expected expression")
            return None
    
    def map_expression(self, line: int, col: int) -> MapExpression:
        """Parse list comprehension: [map items with x action x * 2] or [map items with x where x > 0 action x * 2]"""
        
        # Parse the list expression
        list_expr = self.expression()
        
        if not self.check("WITH"):
            self.error("Expected 'with' in map expression")
        self.advance()
        
        if not self.check("IDENTIFIER"):
            self.error("Expected variable name after 'with'")
        
        var_token = self.advance()
        variable = Identifier(var_token.lexeme, var_token.line, var_token.col)
        
        # Check for optional WHERE clause
        filter_cond = None
        if self.check("WHERE"):
            self.advance()
            filter_cond = self.expression()
        
        if not self.check("ACTION"):
            self.error("Expected 'action' in map expression")
        self.advance()
        
        transform = self.expression()
        
        if not self.check("RIGHT_BRACKET"):
            self.error("Expected ']' after map expression")
        self.advance()
        
        return MapExpression(list_expr, variable, transform, filter_cond, line, col)
    
    def _parse_string_interpolation(self, text: str, line: int, col: int) -> List[Union[str, ASTNode]]:
        """Parse string interpolation: "Hello {name}!" -> ["Hello ", Identifier("name"), "!"]"""
        parts = []
        i = 0
        
        while i < len(text):
            if text[i] == '{' and i + 1 < len(text) and text[i + 1] != '{':
                # Start of interpolation
                j = i + 1
                brace_count = 1
                
                while j < len(text) and brace_count > 0:
                    if text[j] == '{':
                        brace_count += 1
                    elif text[j] == '}':
                        brace_count -= 1
                    j += 1
                
                if brace_count == 0:
                    # Parse expression inside braces
                    expr_text = text[i+1:j-1].strip()
                    # Create a mini-lexer and parser for the expression
                    try:
                        lexer = Lexer(expr_text, "<interpolation>")
                        tokens = lexer.scan_tokens()
                        if not lexer.has_errors():
                            parser = Parser(tokens, expr_text)
                            expr = parser.expression()
                            if expr and not parser.has_errors():
                                parts.append(expr)
                            else:
                                parts.append(f"{{{expr_text}}}")  # Fallback to literal
                        else:
                            parts.append(f"{{{expr_text}}}")  # Fallback to literal
                    except:
                        parts.append(f"{{{expr_text}}}")  # Fallback to literal
                    
                    i = j
                    continue
            
            # Regular character
            start = i
            while i < len(text) and not (text[i] == '{' and i + 1 < len(text) and text[i + 1] != '{'):
                i += 1
            
            if start < i:
                parts.append(text[start:i])
        
        return parts
    
    def block(self) -> List[ASTNode]:
        statements = []
        
        while not (self.check("END") or self.check("ELSE") or 
                  self.check("RESCUE") or self.is_at_end()):
            stmt = self.statement()
            if stmt:
                statements.append(stmt)
        
        return statements
    
    def check(self, type_name: str) -> bool:
        if self.is_at_end():
            return False
        return self.peek().type.name == type_name
    
    def advance(self):
        if not self.is_at_end():
            self.current += 1
        return self.previous()
    
    def is_at_end(self) -> bool:
        return self.peek().type.name == "EOF"
    
    def peek(self):
        return self.tokens[self.current]
    
    def peek_next(self):
        if self.current + 1 < len(self.tokens):
            return self.tokens[self.current + 1]
        return self.tokens[-1]
    
    def peek_type(self) -> str:
        return self.peek().type.name
    
    def peek_next_type(self) -> str:
        return self.peek_next().type.name
    
    def previous(self, offset: int = 1):
        return self.tokens[self.current - offset]
    
    def previous_location(self):
        token = self.previous()
        return token.line, token.col
    
    def current_location(self):
        token = self.peek()
        return token.line, token.col
    
    def error(self, message: str):
        token = self.peek()
        raise ParserError(message, token.line, token.col, self.source)
    
    def synchronize(self):
        while not self.is_at_end():
            if self.previous().type.name == "END":
                return
            
            if self.peek().type.name in [
                "GLOW", "SAY", "SET", "ASK", "IF", "REPEAT", 
                "FOR", "WHILE", "TRY", "ACTION", "RETURN", "END", 
                "STOP", "CONTINUE", "WAIT", "DO", "IMPORT", "RESCUE"
            ]:
                return
            
            self.advance()
    
    def has_errors(self) -> bool:
        return len(self.errors) > 0


# ============================================================================
# 6. ENVIRONMENT (SAME)
# ============================================================================

class Environment:
    """Runtime environment for variable storage"""
    
    def __init__(self, parent: Optional['Environment'] = None):
        self.parent = parent
        self.values: Dict[str, Any] = {}
        self.constants: Set[str] = set()
    
    def define(self, name: str, value: Any, is_const: bool = False):
        self.values[name] = value
        if is_const:
            self.constants.add(name)
    
    def assign(self, name: str, value: Any):
        if name in self.values:
            if name in self.constants:
                raise RuntimeError(f"Cannot reassign constant '{name}'")
            self.values[name] = value
            return
        
        if self.parent:
            self.parent.assign(name, value)
            return
        
        raise RuntimeError(f"Undefined variable '{name}'")
    
    def get(self, name: str) -> Any:
        if name in self.values:
            return self.values[name]
        
        if self.parent:
            return self.parent.get(name)
        
        raise RuntimeError(f"Undefined variable '{name}'")
    
    def get_at(self, distance: int, name: str) -> Any:
        env = self
        for _ in range(distance):
            env = env.parent
        return env.values.get(name)
    
    def assign_at(self, distance: int, name: str, value: Any):
        env = self
        for _ in range(distance):
            env = env.parent
        env.values[name] = value
    
    def ancestor(self, distance: int) -> 'Environment':
        env = self
        for _ in range(distance):
            env = env.parent
        return env
    
    def contains(self, name: str) -> bool:
        if name in self.values:
            return True
        if self.parent:
            return self.parent.contains(name)
        return False
    
    def copy(self) -> 'Environment':
        new_env = Environment(self.parent)
        new_env.values = self.values.copy()
        new_env.constants = self.constants.copy()
        return new_env


# ============================================================================
# 7. BOUND METHOD CLASS (NEW - FIXES METHOD BINDING)
# ============================================================================

class BoundMethod:
    """Proper bound method class for object.method() calls"""
    def __init__(self, obj: Any, method_name: str, method_func: Callable):
        self.obj = obj
        self.method_name = method_name
        self.method_func = method_func
    
    def __call__(self, *args):
        return self.method_func(self.obj, *args)
    
    def __repr__(self):
        return f"<bound method '{self.method_name}'>"
    
    def __str__(self):
        return f"<method '{self.method_name}'>"


# ============================================================================
# 8. BUILTIN FUNCTIONS AND TYPES (ENHANCED)
# ============================================================================

class ReturnException(Exception):
    def __init__(self, value):
        self.value = value


class GlowFunction:
    def __init__(self, declaration, closure):
        self.declaration = declaration
        self.closure = closure
        self.name = declaration.name.name
    
    def __call__(self, *args):
        interpreter = Interpreter()
        
        env = Environment(self.closure)
        
        for i, param in enumerate(self.declaration.params):
            if i < len(args):
                env.define(param.name, args[i])
            else:
                env.define(param.name, None)
        
        try:
            interpreter.execute_block(self.declaration.body, env)
        except ReturnException as ret:
            return ret.value
        
        return None
    
    def __repr__(self):
        return f"<function {self.name}>"


class StringMethods:
    @staticmethod
    def uppercase(s: str) -> str:
        return s.upper()
    
    @staticmethod
    def lowercase(s: str) -> str:
        return s.lower()
    
    @staticmethod
    def capitalize(s: str) -> str:
        return s.capitalize()
    
    @staticmethod
    def title(s: str) -> str:
        return s.title()
    
    @staticmethod
    def trim(s: str) -> str:
        return s.strip()
    
    @staticmethod
    def trim_start(s: str) -> str:
        return s.lstrip()
    
    @staticmethod
    def trim_end(s: str) -> str:
        return s.rstrip()
    
    @staticmethod
    def replace(s: str, old: str, new: str) -> str:
        return s.replace(old, new)
    
    @staticmethod
    def split(s: str, delimiter: str = " ") -> list:
        return s.split(delimiter)
    
    @staticmethod
    def starts_with(s: str, prefix: str) -> bool:
        return s.startswith(prefix)
    
    @staticmethod
    def ends_with(s: str, suffix: str) -> bool:
        return s.endswith(suffix)
    
    @staticmethod
    def contains(s: str, substring: str) -> bool:
        return substring in s
    
    @staticmethod
    def slice(s: str, start: int, end: int = None, step: int = 1) -> str:
        if end is None:
            end = len(s)
        return s[start:end:step]
    
    @staticmethod
    def repeat(s: str, times: int) -> str:
        return s * times
    
    @staticmethod
    def pad_start(s: str, length: int, char: str = " ") -> str:
        return s.rjust(length, char)
    
    @staticmethod
    def pad_end(s: str, length: int, char: str = " ") -> str:
        return s.ljust(length, char)
    
    @staticmethod
    def find(s: str, substring: str) -> int:
        return s.find(substring)
    
    @staticmethod
    def count(s: str, substring: str) -> int:
        return s.count(substring)


class ListMethods:
    @staticmethod
    def push(lst: list, *items) -> list:
        lst.extend(items)
        return lst
    
    @staticmethod
    def pop(lst: list, index: int = -1):
        if len(lst) == 0:
            raise RuntimeError("Cannot pop from empty list")
        return lst.pop(index)
    
    @staticmethod
    def insert(lst: list, index: int, item) -> list:
        lst.insert(index, item)
        return lst
    
    @staticmethod
    def remove(lst: list, item) -> list:
        lst.remove(item)
        return lst
    
    @staticmethod
    def index_of(lst: list, item) -> int:
        try:
            return lst.index(item)
        except ValueError:
            return -1
    
    @staticmethod
    def contains(lst: list, item) -> bool:
        return item in lst
    
    @staticmethod
    def slice(lst: list, start: int, end: int = None, step: int = 1) -> list:
        if end is None:
            end = len(lst)
        return lst[start:end:step]
    
    @staticmethod
    def sort(lst: list, reverse: bool = False) -> list:
        sorted_list = sorted(lst)
        if reverse:
            sorted_list.reverse()
        return sorted_list
    
    @staticmethod
    def reverse(lst: list) -> list:
        lst.reverse()
        return lst
    
    @staticmethod
    def join(lst: list, separator: str = "") -> str:
        return separator.join(str(item) for item in lst)
    
    @staticmethod
    def filter(lst: list, condition_func: Callable) -> list:
        return [item for item in lst if condition_func(item)]
    
    @staticmethod
    def map(lst: list, transform_func: Callable) -> list:
        return [transform_func(item) for item in lst]


class Builtins:
    """Collection of built-in functions with improved error messages"""
    
    @staticmethod
    def _friendly_type(value: Any) -> str:
        """Convert Python types to kid-friendly names"""
        if value is None:
            return "nothing"
        elif isinstance(value, bool):
            return "true/false"
        elif isinstance(value, (int, float)):
            return "number"
        elif isinstance(value, str):
            return "text"
        elif isinstance(value, list):
            return "list"
        elif isinstance(value, dict):
            return "object"
        elif callable(value):
            return "action"
        else:
            return "thing"
    
    @staticmethod
    def _type_check(value: Any, expected_type: type, name: str):
        if not isinstance(value, expected_type):
            friendly_expected = Builtins._friendly_type(expected_type())
            friendly_got = Builtins._friendly_type(value)
            raise TypeError(
                f"{name} needs {friendly_expected}, but you gave it {friendly_got}"
            )
    
    @staticmethod
    def print(*args):
        """Print values with kid-friendly formatting"""
        formatted = []
        for arg in args:
            if arg is None:
                formatted.append("null")
            elif isinstance(arg, bool):
                formatted.append("true" if arg else "false")
            elif isinstance(arg, float) and arg.is_integer():
                formatted.append(str(int(arg)))
            else:
                formatted.append(str(arg))
        print(*formatted)
        return None
    
    @staticmethod
    def input(prompt: str = "") -> str:
        """Get user input"""
        Builtins._type_check(prompt, str, "input")
        return input(prompt)
    
    @staticmethod
    def len(value) -> int:
        """Get length of string, list, or object"""
        if isinstance(value, str) or isinstance(value, list):
            return len(value)
        elif isinstance(value, dict):
            return len(value)
        raise TypeError(f"len() needs text, list, or object, but got {Builtins._friendly_type(value)}")
    
    @staticmethod
    def length(value) -> int:
        """Kid-friendly alias for len"""
        return Builtins.len(value)
    
    @staticmethod
    def to_string(value: Any) -> str:
        """Convert value to string"""
        if value is None:
            return "null"
        elif isinstance(value, bool):
            return "true" if value else "false"
        elif isinstance(value, float) and value.is_integer():
            return str(int(value))
        return str(value)
    
    @staticmethod
    def to_number(value: Any) -> float:
        """Convert value to number"""
        if isinstance(value, (int, float)):
            return float(value)
        elif isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                raise TypeError(f"Can't turn '{value}' into a number (is it text instead?)")
        elif isinstance(value, bool):
            return 1.0 if value else 0.0
        elif value is None:
            return 0.0
        raise TypeError(f"Can't turn {Builtins._friendly_type(value)} into a number")
    
    @staticmethod
    def list_new(size: int = 0) -> list:
        """Create new list with given size"""
        Builtins._type_check(size, (int, float), "list_new")
        return [None] * int(size)
    
    @staticmethod
    def list_push(lst: list, value: Any) -> list:
        """Push value to list"""
        Builtins._type_check(lst, list, "list_push")
        lst.append(value)
        return lst
    
    @staticmethod
    def list_pop(lst: list):
        """Pop value from list"""
        Builtins._type_check(lst, list, "list_pop")
        if len(lst) == 0:
            raise RuntimeError("Can't pop from empty list - it's already empty!")
        return lst.pop()
    
    @staticmethod
    def list_get(lst: list, index: int):
        """Get element at index"""
        Builtins._type_check(lst, list, "list_get")
        Builtins._type_check(index, (int, float), "list_get")
        idx = int(index)
        
        # Handle negative indices
        if idx < 0:
            idx = len(lst) + idx
        
        if idx < 0 or idx >= len(lst):
            raise RuntimeError(
                f"Index {idx} is out of bounds. "
                f"The list has {len(lst)} items (indices 0 to {len(lst)-1})"
            )
        return lst[idx]
    
    @staticmethod
    def list_set(lst: list, index: int, value: Any) -> list:
        """Set element at index"""
        Builtins._type_check(lst, list, "list_set")
        Builtins._type_check(index, (int, float), "list_set")
        idx = int(index)
        
        # Handle negative indices
        if idx < 0:
            idx = len(lst) + idx
        
        if idx < 0 or idx >= len(lst):
            raise RuntimeError(
                f"Index {idx} is out of bounds. "
                f"The list has {len(lst)} items (indices 0 to {len(lst)-1})"
            )
        lst[idx] = value
        return lst
    
    @staticmethod
    def random() -> float:
        """Random number between 0 and 1"""
        return random.random()
    
    @staticmethod
    def random_range(min_val: int, max_val: int) -> int:
        """Random integer between min and max (inclusive)"""
        Builtins._type_check(min_val, (int, float), "random_range")
        Builtins._type_check(max_val, (int, float), "random_range")
        return random.randint(int(min_val), int(max_val))
    
    @staticmethod
    def choose(lst: list) -> Any:
        """Randomly choose an element from a list"""
        Builtins._type_check(lst, list, "choose")
        if len(lst) == 0:
            raise RuntimeError("Can't choose from empty list - add some items first!")
        return random.choice(lst)
    
    @staticmethod
    def time() -> float:
        """Current time in seconds"""
        return time.time()
    
    @staticmethod
    def sleep(seconds: float):
        """Sleep for given seconds"""
        Builtins._type_check(seconds, (int, float), "sleep")
        time.sleep(seconds)
        return None
    
    @staticmethod
    def wait(seconds: float):
        """Kid-friendly alias for sleep"""
        return Builtins.sleep(seconds)
    
    @staticmethod
    def sqrt(x: float) -> float:
        """Square root"""
        Builtins._type_check(x, (int, float), "sqrt")
        if x < 0:
            raise RuntimeError("Can't take square root of negative number")
        return math.sqrt(x)
    
    @staticmethod
    def abs(x: float) -> float:
        """Absolute value"""
        Builtins._type_check(x, (int, float), "abs")
        return abs(x)
    
    @staticmethod
    def round(x: float, decimals: int = 0) -> float:
        """Round number"""
        Builtins._type_check(x, (int, float), "round")
        Builtins._type_check(decimals, (int, float), "round")
        return round(x, int(decimals))
    
    @staticmethod
    def min(*args):
        """Minimum of values"""
        if len(args) == 0:
            raise RuntimeError("min() needs at least one number")
        return min(args)
    
    @staticmethod
    def max(*args):
        """Maximum of values"""
        if len(args) == 0:
            raise RuntimeError("max() needs at least one number")
        return max(args)
    
    @staticmethod
    def sum(lst: list) -> float:
        """Sum of list"""
        Builtins._type_check(lst, list, "sum")
        total = 0.0
        for item in lst:
            if isinstance(item, (int, float)):
                total += item
        return total
    
    @staticmethod
    def range(start: int, end: int, step: int = 1) -> list:
        """Generate range as list"""
        Builtins._type_check(start, (int, float), "range")
        Builtins._type_check(end, (int, float), "range")
        Builtins._type_check(step, (int, float), "range")
        return list(range(int(start), int(end), int(step)))
    
    @staticmethod
    def map_type(value: Any) -> str:
        """Get type of value as string"""
        return Builtins._friendly_type(value)
    
    @staticmethod
    def what_is(value: Any) -> str:
        """Kid-friendly alias for type"""
        return Builtins.map_type(value)
    
    @staticmethod
    def read_file(path: str) -> str:
        """Read entire file as string"""
        Builtins._type_check(path, str, "read_file")
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            raise RuntimeError(f"Couldn't read file '{path}': {str(e)}")
    
    @staticmethod
    def write_file(path: str, contents: Any) -> None:
        """Write string to file"""
        Builtins._type_check(path, str, "write_file")
        try:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(str(contents))
        except Exception as e:
            raise RuntimeError(f"Couldn't write file '{path}': {str(e)}")
    
    @staticmethod
    def json_parse(s: str) -> Any:
        """Parse JSON string into GLOW value"""
        Builtins._type_check(s, str, "json_parse")
        try:
            return json.loads(s)
        except Exception as e:
            raise RuntimeError(f"Invalid JSON: {str(e)}")
    
    @staticmethod
    def json_stringify(value: Any) -> str:
        """Convert GLOW value to JSON string"""
        if isinstance(value, dict):
            return json.dumps(value)
        elif isinstance(value, list):
            return json.dumps(value)
        elif value is None:
            return "null"
        elif isinstance(value, bool):
            return "true" if value else "false"
        elif isinstance(value, (int, float)):
            return str(value)
        elif isinstance(value, str):
            return json.dumps(value)
        else:
            raise TypeError(f"Can't turn {Builtins._friendly_type(value)} into JSON")
    
    @staticmethod
    def clone(value: Any) -> Any:
        """Create a deep copy of a value"""
        if isinstance(value, dict):
            return {k: Builtins.clone(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [Builtins.clone(item) for item in value]
        else:
            return value
    
    # String methods
    @staticmethod
    def uppercase(s: str) -> str:
        Builtins._type_check(s, str, "uppercase")
        return StringMethods.uppercase(s)
    
    @staticmethod
    def lowercase(s: str) -> str:
        Builtins._type_check(s, str, "lowercase")
        return StringMethods.lowercase(s)
    
    @staticmethod
    def capitalize(s: str) -> str:
        Builtins._type_check(s, str, "capitalize")
        return StringMethods.capitalize(s)
    
    @staticmethod
    def trim(s: str) -> str:
        Builtins._type_check(s, str, "trim")
        return StringMethods.trim(s)
    
    @staticmethod
    def split(s: str, delimiter: str = " ") -> list:
        Builtins._type_check(s, str, "split")
        Builtins._type_check(delimiter, str, "split")
        return StringMethods.split(s, delimiter)
    
    @staticmethod
    def replace(s: str, old: str, new: str) -> str:
        Builtins._type_check(s, str, "replace")
        Builtins._type_check(old, str, "replace")
        Builtins._type_check(new, str, "replace")
        return StringMethods.replace(s, old, new)
    
    # List methods
    @staticmethod
    def list_insert(lst: list, index: int, item) -> list:
        Builtins._type_check(lst, list, "list_insert")
        Builtins._type_check(index, (int, float), "list_insert")
        return ListMethods.insert(lst, int(index), item)
    
    @staticmethod
    def list_remove(lst: list, item) -> list:
        Builtins._type_check(lst, list, "list_remove")
        return ListMethods.remove(lst, item)
    
    @staticmethod
    def list_index_of(lst: list, item) -> int:
        Builtins._type_check(lst, list, "list_index_of")
        return ListMethods.index_of(lst, item)
    
    @staticmethod
    def list_sort(lst: list, reverse: bool = False) -> list:
        Builtins._type_check(lst, list, "list_sort")
        return ListMethods.sort(lst, reverse)
    
    @staticmethod
    def list_reverse(lst: list) -> list:
        Builtins._type_check(lst, list, "list_reverse")
        return ListMethods.reverse(lst)
    
    # PHASE 2: HTTP REQUESTS
    @staticmethod
    def fetch(url: str, options: dict = None) -> dict:
        """Make HTTP request"""
        Builtins._type_check(url, str, "fetch")
        
        try:
            req = urllib.request.Request(url)
            
            if options:
                Builtins._type_check(options, dict, "fetch options")
                
                if 'method' in options:
                    req.method = options['method'].upper()
                
                if 'headers' in options:
                    for key, value in options['headers'].items():
                        req.add_header(key, value)
                
                if 'body' in options and req.method in ['POST', 'PUT', 'PATCH']:
                    data = str(options['body']).encode('utf-8')
                    req.data = data
                    req.add_header('Content-Type', 'application/json')
            
            with urllib.request.urlopen(req, timeout=10) as response:
                body = response.read().decode('utf-8')
                
                # Try to parse as JSON
                try:
                    parsed_body = json.loads(body)
                except:
                    parsed_body = body
                
                return {
                    'ok': True,
                    'status': response.status,
                    'body': parsed_body,
                    'headers': dict(response.headers)
                }
                
        except urllib.error.HTTPError as e:
            return {
                'ok': False,
                'status': e.code,
                'error': str(e),
                'body': e.read().decode('utf-8') if hasattr(e, 'read') else None
            }
        except Exception as e:
            return {
                'ok': False,
                'status': 0,
                'error': str(e)
            }
    
    @staticmethod
    def fetch_json(url: str) -> Any:
        """Fetch and parse JSON"""
        response = Builtins.fetch(url)
        if response['ok']:
            return response['body']
        else:
            raise RuntimeError(f"HTTP Error {response['status']}: {response.get('error', 'Unknown error')}")


# Core builtins
CORE_BUILTINS = {
    "print": Builtins.print,
    "input": Builtins.input,
    "len": Builtins.len,
    "length": Builtins.length,
    "to_string": Builtins.to_string,
    "to_number": Builtins.to_number,
    "list_new": Builtins.list_new,
    "list_push": Builtins.list_push,
    "list_pop": Builtins.list_pop,
    "list_get": Builtins.list_get,
    "list_set": Builtins.list_set,
    "list_insert": Builtins.list_insert,
    "list_remove": Builtins.list_remove,
    "list_index_of": Builtins.list_index_of,
    "list_sort": Builtins.list_sort,
    "list_reverse": Builtins.list_reverse,
    "random": Builtins.random,
    "random_range": Builtins.random_range,
    "choose": Builtins.choose,
    "time": Builtins.time,
    "sleep": Builtins.sleep,
    "wait": Builtins.wait,
    "sqrt": Builtins.sqrt,
    "abs": Builtins.abs,
    "round": Builtins.round,
    "min": Builtins.min,
    "max": Builtins.max,
    "sum": Builtins.sum,
    "range": Builtins.range,
    "type": Builtins.map_type,
    "what_is": Builtins.what_is,
    "clone": Builtins.clone,
    "uppercase": Builtins.uppercase,
    "lowercase": Builtins.lowercase,
    "capitalize": Builtins.capitalize,
    "trim": Builtins.trim,
    "split": Builtins.split,
    "replace": Builtins.replace,
    "fetch": Builtins.fetch,
    "fetch_json": Builtins.fetch_json,
}

# Permission-gated builtins
PERMISSION_BUILTINS = {
    "read_file": Builtins.read_file,
    "write_file": Builtins.write_file,
    "json_parse": Builtins.json_parse,
    "json_stringify": Builtins.json_stringify,
}


# ============================================================================
# 9. INTERPRETER (COMPLETELY FIXED)
# ============================================================================

class BreakException(Exception):
    pass


class ContinueException(Exception):
    pass


class Interpreter:
    """Tree-walking interpreter for GLOW - FIXED VERSION"""
    
    def __init__(self, kid_mode: bool = False, allow_files: bool = False, allow_http: bool = False):
        self.globals = Environment()
        self.environment = self.globals
        self.locals = {}
        self.kid_mode = kid_mode
        self.allow_files = allow_files
        self.allow_http = allow_http
        self.loaded_modules: Dict[str, Dict[str, Any]] = {}
        
        # Add core builtins
        for name, func in CORE_BUILTINS.items():
            if name in ['fetch', 'fetch_json'] and not allow_http:
                continue  # Skip HTTP functions if not allowed
            self.globals.define(name, func)
        
        # Add permission-gated builtins if allowed
        if allow_files:
            for name, func in PERMISSION_BUILTINS.items():
                self.globals.define(name, func)
        
        # Setup method tables
        self._setup_method_tables()
    
    def _setup_method_tables(self):
        """Setup string and list method tables"""
        self.string_methods = {
            "uppercase": StringMethods.uppercase,
            "lowercase": StringMethods.lowercase,
            "capitalize": StringMethods.capitalize,
            "title": StringMethods.title,
            "trim": StringMethods.trim,
            "trim_start": StringMethods.trim_start,
            "trim_end": StringMethods.trim_end,
            "replace": StringMethods.replace,
            "split": StringMethods.split,
            "starts_with": StringMethods.starts_with,
            "ends_with": StringMethods.ends_with,
            "contains": StringMethods.contains,
            "slice": StringMethods.slice,
            "repeat": StringMethods.repeat,
            "pad_start": StringMethods.pad_start,
            "pad_end": StringMethods.pad_end,
            "find": StringMethods.find,
            "count": StringMethods.count,
            "length": lambda s: len(s),
        }
        
        self.list_methods = {
            "push": ListMethods.push,
            "pop": ListMethods.pop,
            "insert": ListMethods.insert,
            "remove": ListMethods.remove,
            "index_of": ListMethods.index_of,
            "contains": ListMethods.contains,
            "slice": ListMethods.slice,
            "sort": ListMethods.sort,
            "reverse": ListMethods.reverse,
            "join": ListMethods.join,
            "filter": ListMethods.filter,
            "map": ListMethods.map,
            "length": lambda lst: len(lst),
        }
    
    def interpret(self, ast, module_name: str = "<main>"):
        try:
            result = self.execute(ast)
            return result
        except RuntimeError as e:
            if self.kid_mode:
                original_msg = str(e)
                # Add kid-friendly hints
                if "undefined" in original_msg.lower():
                    e.message = f"{original_msg} (Did you forget to 'set' it first?)"
                elif "division by zero" in original_msg.lower():
                    e.message = f"{original_msg} (You can't divide by zero!)"
                elif "index out of bounds" in original_msg.lower():
                    e.message = f"{original_msg} (The list isn't that long!)"
                elif "can't add" in original_msg.lower():
                    e.message = f"{original_msg} (Try using to_string() to convert to text first!)"
            raise e
    
    def execute(self, node):
        method_name = f'visit_{node.__class__.__name__.lower()}'
        method = getattr(self, method_name, self.generic_visit)
        return method(node)
    
    def generic_visit(self, node):
        raise RuntimeError(f"No visit method for {node.__class__.__name__}")
    
    def visit_program(self, node):
        result = None
        for statement in node.statements:
            result = self.execute(statement)
        return result
    
    def visit_literal(self, node):
        return node.value
    
    def visit_interpolated_string(self, node):
        """Evaluate interpolated string: "Hello {name}!" """
        result_parts = []
        for part in node.parts:
            if isinstance(part, str):
                result_parts.append(part)
            else:
                # It's an AST node, evaluate it
                value = self.execute(part)
                result_parts.append(str(value))
        return ''.join(result_parts)
    
    def visit_identifier(self, node):
        return self.environment.get(node.name)
    
    def visit_binary_op(self, node):
        left = self.execute(node.left)
        right = self.execute(node.right)
        
        # Get kid-friendly type names
        left_type = self._friendly_type(left)
        right_type = self._friendly_type(right)
        
        if node.operator == "+":
            if isinstance(left, str) or isinstance(right, str):
                return str(left) + str(right)
            elif isinstance(left, (int, float)) and isinstance(right, (int, float)):
                return left + right
            elif isinstance(left, list) and isinstance(right, list):
                return left + right
            else:
                raise TypeError(f"Can't add {left_type} and {right_type} together")
        
        elif node.operator == "-":
            left_num = self._to_number(left, "left side of -")
            right_num = self._to_number(right, "right side of -")
            return left_num - right_num
        
        elif node.operator == "*":
            left_num = self._to_number(left, "left side of *")
            right_num = self._to_number(right, "right side of *")
            return left_num * right_num
        
        elif node.operator == "/":
            left_num = self._to_number(left, "left side of /")
            right_num = self._to_number(right, "right side of /")
            if right_num == 0:
                raise RuntimeError("Division by zero")
            return left_num / right_num
        
        elif node.operator == "%":
            left_num = self._to_number(left, "left side of %")
            right_num = self._to_number(right, "right side of %")
            if right_num == 0:
                raise RuntimeError("Modulo by zero")
            return left_num % right_num
        
        elif node.operator in ["and", "or"]:
            left_bool = self._is_truthy(left)
            if node.operator == "and":
                if not left_bool:
                    return False
                return self._is_truthy(right)
            else:
                if left_bool:
                    return True
                return self._is_truthy(right)
        
        elif node.operator in ["is", "is not"]:
            is_equal = self._is_equal(left, right)
            return is_equal if node.operator == "is" else not is_equal
        
        elif node.operator in ["<", ">", "<=", ">=", 
                              "is bigger than", "is smaller than",
                              "is bigger or equal to", "is smaller or equal to"]:
            op = node.operator
            if op == "is bigger than": op = ">"
            elif op == "is smaller than": op = "<"
            elif op == "is bigger or equal to": op = ">="
            elif op == "is smaller or equal to": op = "<="
            
            if isinstance(left, (int, float)) and isinstance(right, (int, float)):
                if op == "<": return left < right
                elif op == ">": return left > right
                elif op == "<=": return left <= right
                elif op == ">=": return left >= right
            elif isinstance(left, str) and isinstance(right, str):
                if op == "<": return left < right
                elif op == ">": return left > right
                elif op == "<=": return left <= right
                elif op == ">=": return left >= right
            else:
                raise TypeError(f"Can't compare {left_type} and {right_type}")
        
        raise RuntimeError(f"Unknown operator: {node.operator}")
    
    def visit_unary_op(self, node):
        right = self.execute(node.right)
        
        if node.operator == "-":
            right_num = self._to_number(right, "value after -")
            return -right_num
        elif node.operator == "not":
            return not self._is_truthy(right)
        
        raise RuntimeError(f"Unknown unary operator: {node.operator}")
    
    def visit_in_expression(self, node):
        """Evaluate 'item in container' expression"""
        item = self.execute(node.item)
        container = self.execute(node.container)
        
        if isinstance(container, list):
            return item in container
        elif isinstance(container, str):
            if isinstance(item, str):
                return item in container
            else:
                return str(item) in container
        elif isinstance(container, dict):
            return item in container
        else:
            raise TypeError(f"Can't check if something is in {self._friendly_type(container)}")
    
    def visit_set_statement(self, node):
        value = self.execute(node.value)
        try:
            self.environment.assign(node.name.name, value)
        except RuntimeError:
            # Variable doesn't exist, so define it
            self.environment.define(node.name.name, value)
        return value
    
    def visit_set_property_statement(self, node):
        """Handle obj.property = value"""
        obj = self.execute(node.object_expr)
        value = self.execute(node.value)
        
        if isinstance(obj, dict):
            obj[node.property_name] = value
        else:
            raise TypeError(f"Can't set property on {self._friendly_type(obj)}")
        return value
    
    def visit_set_property(self, node):
        """Handle obj[index] = value"""
        obj = self.execute(node.object_expr)
        index_val = self.execute(node.index)
        value = self.execute(node.value)
        
        if isinstance(obj, list):
            idx = self._to_number(index_val, "array index")
            idx_int = int(idx)
            
            # Handle negative indices
            if idx_int < 0:
                idx_int = len(obj) + idx_int
            
            if idx_int < 0 or idx_int >= len(obj):
                # Allow extending the list if index is exactly at the end
                if idx_int == len(obj):
                    obj.append(value)
                else:
                    raise RuntimeError(
                        f"Index {idx_int} out of bounds. "
                        f"The list has {len(obj)} items (indices 0 to {len(obj)-1})"
                    )
            else:
                obj[idx_int] = value
        elif isinstance(obj, dict):
            # Convert index to string for dict keys
            obj[str(index_val)] = value
        else:
            raise TypeError(f"Can't set index on {self._friendly_type(obj)}")
        
        return value
    
    def visit_index_expression(self, node):
        """Handle obj[index]"""
        obj = self.execute(node.object_expr)
        index_val = self.execute(node.index)
        
        if isinstance(obj, list):
            idx = self._to_number(index_val, "array index")
            idx_int = int(idx)
            
            # Handle negative indices
            if idx_int < 0:
                idx_int = len(obj) + idx_int
            
            if idx_int < 0 or idx_int >= len(obj):
                raise RuntimeError(
                    f"Index {idx_int} out of bounds. "
                    f"The list has {len(obj)} items (indices 0 to {len(obj)-1})"
                )
            
            return obj[idx_int]
        elif isinstance(obj, str):
            idx = self._to_number(index_val, "string index")
            idx_int = int(idx)
            
            # Handle negative indices
            if idx_int < 0:
                idx_int = len(obj) + idx_int
            
            if idx_int < 0 or idx_int >= len(obj):
                raise RuntimeError(
                    f"Index {idx_int} out of bounds. "
                    f"The string has {len(obj)} characters (indices 0 to {len(obj)-1})"
                )
            
            return obj[idx_int]
        elif isinstance(obj, dict):
            # Try to use as string key
            key = str(index_val)
            if key in obj:
                return obj[key]
            else:
                return None
        else:
            raise TypeError(f"Can't index {self._friendly_type(obj)}")
    
    def visit_slice_expression(self, node):
        """Handle obj[start:end:step]"""
        obj = self.execute(node.object_expr)
        start_val = self.execute(node.start) if node.start else None
        end_val = self.execute(node.end) if node.end else None
        step_val = self.execute(node.step) if node.step else None
        
        # Convert to integers
        start = int(self._to_number(start_val, "slice start")) if start_val else None
        end = int(self._to_number(end_val, "slice end")) if end_val else None
        step = int(self._to_number(step_val, "slice step")) if step_val else 1
        
        if isinstance(obj, list):
            return obj[start:end:step]
        elif isinstance(obj, str):
            return obj[start:end:step]
        else:
            raise TypeError(f"Can't slice {self._friendly_type(obj)}")
    
    def visit_map_expression(self, node):
        """Handle list comprehensions: [map items with x action x * 2]"""
        list_expr = self.execute(node.list_expr)
        
        if not isinstance(list_expr, list):
            raise TypeError(f"Can't map over {self._friendly_type(list_expr)}")
        
        result = []
        for item in list_expr:
            # Create environment for the variable
            env = Environment(self.environment)
            env.define(node.variable.name, item)
            
            # Check filter condition if present
            if node.filter_cond:
                # Evaluate filter in the new environment
                old_env = self.environment
                self.environment = env
                try:
                    filter_result = self.execute(node.filter_cond)
                    if not self._is_truthy(filter_result):
                        continue  # Skip this item
                finally:
                    self.environment = old_env
            
            # Apply transform
            old_env = self.environment
            self.environment = env
            try:
                transformed = self.execute(node.transform)
                result.append(transformed)
            finally:
                self.environment = old_env
        
        return result
    
    def visit_glow_statement(self, node):
        value = self.execute(node.expression)
        emoji = "✨" if self.kid_mode else ""
        if value is None:
            print(f"{emoji} null")
        elif isinstance(value, bool):
            print(f"{emoji} " + ("true" if value else "false"))
        else:
            print(f"{emoji} {value}")
        return None
    
    def visit_wait_statement(self, node):
        seconds = self.execute(node.seconds)
        seconds_num = self._to_number(seconds, "wait time")
        if self.kid_mode:
            print(f"⏰ Waiting {seconds_num} seconds...")
        time.sleep(seconds_num)
        return None
    
    def visit_ask_statement(self, node):
        response = input(node.prompt + " ")
        self.environment.define(node.name.name, response)
        return response
    
    def visit_if_statement(self, node):
        condition = self.execute(node.condition)
        
        if self._is_truthy(condition):
            self.execute_block(node.then_branch, Environment(self.environment))
        elif node.else_branch:
            self.execute_block(node.else_branch, Environment(self.environment))
        
        return None
    
    def visit_while_statement(self, node):
        MAX_LOOPS = 1000000 if not self.kid_mode else 10000
        loop_count = 0
        
        while self._is_truthy(self.execute(node.condition)):
            if loop_count >= MAX_LOOPS:
                print("⚠️  Safety limit reached: stopped after too many loops")
                break
                
            try:
                self.execute_block(node.body, Environment(self.environment))
            except BreakException:
                break
            except ContinueException:
                continue
            
            loop_count += 1
        
        return None
    
    def visit_try_statement(self, node):
        try:
            self.execute_block(node.try_body, Environment(self.environment))
        except (GlowError, RuntimeError, TypeError, ImportError) as e:
            if node.error_var:
                error_env = Environment(self.environment)
                error_env.define(node.error_var.name, str(e))
                self.execute_block(node.rescue_body, error_env)
            else:
                self.execute_block(node.rescue_body, Environment(self.environment))
        except Exception as e:
            if self.kid_mode:
                kid_error = RuntimeError(f"Something went wrong: {str(e)}", node.line, node.col)
                if node.error_var:
                    error_env = Environment(self.environment)
                    error_env.define(node.error_var.name, str(kid_error))
                    self.execute_block(node.rescue_body, error_env)
                else:
                    self.execute_block(node.rescue_body, Environment(self.environment))
            else:
                raise
        
        return None
    
    def visit_repeat_statement(self, node):
        MAX_LOOPS = 1000000 if not self.kid_mode else 10000
        
        if node.is_forever:
            loop_count = 0
            while True:
                if loop_count >= MAX_LOOPS:
                    print("⚠️  Safety limit reached: stopped after too many loops")
                    break
                    
                try:
                    self.execute_block(node.body, Environment(self.environment))
                except BreakException:
                    break
                except ContinueException:
                    continue
                
                loop_count += 1
        else:
            count = self.execute(node.count)
            count_num = self._to_number(count, "repeat count")
            count_int = int(count_num)
            
            if count_int > MAX_LOOPS:
                print(f"⚠️  Limiting to {MAX_LOOPS} loops for safety")
                count_int = MAX_LOOPS
            
            for i in range(count_int):
                try:
                    loop_env = Environment(self.environment)
                    loop_env.define("index", i)
                    self.execute_block(node.body, loop_env)
                except BreakException:
                    break
                except ContinueException:
                    continue
        
        return None
    
    def visit_for_each_statement(self, node):
        iterable = self.execute(node.iterable)
        
        if isinstance(iterable, list):
            for i, item in enumerate(iterable):
                loop_env = Environment(self.environment)
                loop_env.define(node.variable.name, item)
                
                if node.index_var:
                    loop_env.define(node.index_var.name, i)
                
                try:
                    self.execute_block(node.body, loop_env)
                except BreakException:
                    break
                except ContinueException:
                    continue
        elif isinstance(iterable, str):
            for i, char in enumerate(iterable):
                loop_env = Environment(self.environment)
                loop_env.define(node.variable.name, char)
                
                if node.index_var:
                    loop_env.define(node.index_var.name, i)
                
                try:
                    self.execute_block(node.body, loop_env)
                except BreakException:
                    break
                except ContinueException:
                    continue
        elif isinstance(iterable, dict):
            items = list(iterable.items())
            for i, (key, value) in enumerate(items):
                loop_env = Environment(self.environment)
                loop_env.define(node.variable.name, {"key": key, "value": value})
                
                if node.index_var:
                    loop_env.define(node.index_var.name, i)
                
                try:
                    self.execute_block(node.body, loop_env)
                except BreakException:
                    break
                except ContinueException:
                    continue
        else:
            raise TypeError(f"Can't loop over {self._friendly_type(iterable)}")
        
        return None
    
    def visit_action_statement(self, node):
        func = GlowFunction(node, self.environment)
        self.environment.define(node.name.name, func)
        return func
    
    def visit_call_expression(self, node):
        callee = self.execute(node.callee)
        
        if not callable(callee):
            raise RuntimeError(f"'{node.callee}' is not something you can call")
        
        args = []
        for arg_node in node.args:
            args.append(self.execute(arg_node))
        
        try:
            return callee(*args)
        except ReturnException as ret:
            return ret.value
        except Exception as e:
            if isinstance(e, (RuntimeError, TypeError)):
                raise e
            raise RuntimeError(f"Error calling '{node.callee}': {str(e)}")
    
    def visit_method_call_expression(self, node):
        """Handle obj.method(arg1, arg2) calls"""
        obj = self.execute(node.object_expr)
        
        # Get the method
        if isinstance(obj, str) and node.method_name in self.string_methods:
            method_func = self.string_methods[node.method_name]
            bound_method = BoundMethod(obj, node.method_name, method_func)
        elif isinstance(obj, list) and node.method_name in self.list_methods:
            method_func = self.list_methods[node.method_name]
            bound_method = BoundMethod(obj, node.method_name, method_func)
        elif isinstance(obj, dict) and node.method_name == "get":
            # Special case for dict.get()
            def dict_get(key, default=None):
                return obj.get(key, default)
            bound_method = BoundMethod(obj, "get", dict_get)
        else:
            raise RuntimeError(f"'{node.method_name}' is not a method of {self._friendly_type(obj)}")
        
        # Evaluate arguments
        args = []
        for arg_node in node.args:
            args.append(self.execute(arg_node))
        
        try:
            return bound_method(*args)
        except Exception as e:
            if isinstance(e, (RuntimeError, TypeError)):
                raise e
            raise RuntimeError(f"Error calling '{node.method_name}': {str(e)}")
    
    def visit_do_call_expression(self, node):
        callee = self.execute(node.callee)
        
        if not callable(callee):
            raise RuntimeError(f"'{node.callee.name}' is not something you can call")
        
        args = []
        for arg_node in node.args:
            args.append(self.execute(arg_node))
        
        try:
            return callee(*args)
        except ReturnException as ret:
            return ret.value
        except Exception as e:
            if isinstance(e, (RuntimeError, TypeError)):
                raise e
            raise RuntimeError(f"Error calling '{node.callee.name}': {str(e)}")
    
    def visit_get_property(self, node):
        obj = self.execute(node.object_expr)
        
        # Return bound method for strings and lists
        if isinstance(obj, str) and node.name in self.string_methods:
            method_func = self.string_methods[node.name]
            return BoundMethod(obj, node.name, method_func)
        elif isinstance(obj, list) and node.name in self.list_methods:
            method_func = self.list_methods[node.name]
            return BoundMethod(obj, node.name, method_func)
        
        # Regular property access
        if isinstance(obj, dict):
            return obj.get(node.name)
        elif isinstance(obj, list):
            if node.name == "length":
                return len(obj)
            else:
                raise TypeError(f"List has no property '{node.name}' (did you mean 'length'?)")
        elif isinstance(obj, str):
            if node.name == "length":
                return len(obj)
            else:
                # Check if it's a string method
                if node.name in self.string_methods:
                    method_func = self.string_methods[node.name]
                    return BoundMethod(obj, node.name, method_func)
                else:
                    raise TypeError(f"String has no property '{node.name}'")
        else:
            raise TypeError(f"Can't get property '{node.name}' of {self._friendly_type(obj)}")
    
    def visit_return_statement(self, node):
        value = None
        if node.value:
            value = self.execute(node.value)
        raise ReturnException(value)
    
    def visit_list_literal(self, node):
        elements = []
        for element_node in node.elements:
            elements.append(self.execute(element_node))
        return elements
    
    def visit_object_literal(self, node):
        obj = {}
        for key, value_node in node.properties.items():
            obj[key] = self.execute(value_node)
        return obj
    
    def visit_import_statement(self, node):
        if node.imports:
            raise ImportError("Named imports not yet implemented. Use 'import \"file.glo\" as name'")
        else:
            path = node.path
            
            if path in self.loaded_modules:
                module_exports = self.loaded_modules[path]
            else:
                try:
                    if not path.endswith('.glo'):
                        path += '.glo'
                    
                    with open(path, 'r', encoding='utf-8') as f:
                        source = f.read()
                    
                    lexer = Lexer(source, path)
                    tokens = lexer.scan_tokens()
                    
                    if lexer.has_errors():
                        errors = '\n'.join(str(e) for e in lexer.errors)
                        raise ImportError(f"Error parsing module '{path}':\n{errors}")
                    
                    parser = Parser(tokens, source)
                    ast = parser.parse()
                    
                    if parser.has_errors():
                        errors = '\n'.join(str(e) for e in parser.errors)
                        raise ImportError(f"Error parsing module '{path}':\n{errors}")
                    
                    module_env = Environment(self.globals)
                    old_env = self.environment
                    self.environment = module_env
                    
                    try:
                        self.execute(ast)
                    finally:
                        self.environment = old_env
                    
                    module_exports = {}
                    for name, value in module_env.values.items():
                        if not name.startswith('_'):
                            module_exports[name] = value
                    
                    self.loaded_modules[path] = module_exports
                    
                except FileNotFoundError:
                    raise ImportError(f"Module not found: '{path}'")
                except Exception as e:
                    raise ImportError(f"Error loading module '{path}': {str(e)}")
            
            alias = node.alias if node.alias else Path(path).stem
            self.environment.define(alias, module_exports)
            return module_exports
    
    def visit_stop_statement(self, node):
        raise BreakException()
    
    def visit_continue_statement(self, node):
        raise ContinueException()
    
    def execute_block(self, statements, env):
        previous = self.environment
        try:
            self.environment = env
            for statement in statements:
                self.execute(statement)
        finally:
            self.environment = previous
    
    def _friendly_type(self, value: Any) -> str:
        """Kid-friendly type names"""
        return Builtins._friendly_type(value)
    
    def _is_truthy(self, value):
        if value is None:
            return False
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value != 0
        if isinstance(value, str):
            return len(value) > 0
        if isinstance(value, list):
            return len(value) > 0
        if isinstance(value, dict):
            return len(value) > 0
        return True
    
    def _is_equal(self, a, b):
        if a is None and b is None:
            return True
        if a is None or b is None:
            return False
        return a == b
    
    def _to_number(self, value, context):
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, bool):
            return 1.0 if value else 0.0
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                raise TypeError(f"Can't turn '{value}' into a number ({context})")
        if value is None:
            return 0.0
        raise TypeError(f"Expected number, got {self._friendly_type(value)} ({context})")


# ============================================================================
# 10. REPL (UPDATED)
# ============================================================================

class GlowREPL:
    """Read-Eval-Print Loop for GLOW"""
    
    def __init__(self, kid_mode: bool = False, allow_files: bool = False, allow_http: bool = False):
        self.interpreter = Interpreter(kid_mode, allow_files, allow_http)
        self.history: List[str] = []
        self.kid_mode = kid_mode
        self.allow_files = allow_files
        self.allow_http = allow_http
        self.multiline_buffer = ""
        self.in_multiline = False
    
    # Setup readline history if available
    if HAVE_READLINE:
        histfile = os.path.join(os.path.expanduser("~"), ".glow_history")
        try:
            readline.read_history_file(histfile)
            readline.set_history_length(1000)
        except FileNotFoundError:
            pass
        
        atexit.register(readline.write_history_file, histfile)
    
    def run(self):
        if self.kid_mode:
            print("🌟 🌈 Welcome to GLOW Kid Mode! 🌟 🌈")
            print("Everything is simpler and friendlier here!")
        else:
            print("🌟 Welcome to GLOW 2.1! 🌟")
            print("Type commands, 'help' for help, or '.exit' to quit.")
        
        if self.allow_files:
            print("📁 File operations are ENABLED")
        if self.allow_http:
            print("🌐 HTTP requests are ENABLED")
        
        print("New features: String interpolation, HTTP requests, Array slicing, List comprehensions!")
        
        while True:
            try:
                prompt = "glow> " if not self.kid_mode else "🌟 > "
                if self.in_multiline:
                    prompt = ".... > "
                
                line = input(prompt).strip()
                
                # Handle REPL commands
                if line.startswith('.'):
                    if self.handle_command(line):
                        continue
                
                # Check for multi-line mode
                if line.endswith('...'):
                    self.multiline_buffer += line[:-3] + "\n"
                    self.in_multiline = True
                    continue
                elif self.in_multiline:
                    self.multiline_buffer += line + "\n"
                    if line == "end":
                        result = self.execute(self.multiline_buffer)
                        if result is not None:
                            print(f"=> {result}")
                        self.multiline_buffer = ""
                        self.in_multiline = False
                    continue
                
                self.history.append(line)
                
                if line:
                    result = self.execute(line)
                    if result is not None:
                        print(f"=> {result}")
                        
            except EOFError:
                print("\n✨ Goodbye! ✨")
                break
            except KeyboardInterrupt:
                print("\n(Interrupted)")
                self.multiline_buffer = ""
                self.in_multiline = False
                continue
    
    def handle_command(self, line: str) -> bool:
        cmd = line[1:].strip()
        
        if cmd == "exit" or cmd == "quit":
            print("✨ Goodbye! ✨")
            sys.exit(0)
        elif cmd == "help":
            self.show_help()
            return True
        elif cmd == "clear":
            self.history.clear()
            os.system('cls' if os.name == 'nt' else 'clear')
            print("History cleared.")
            return True
        elif cmd == "history":
            self.show_history()
            return True
        elif cmd.startswith("load "):
            filename = cmd[5:].strip().strip('"\'')
            try:
                with open(filename, 'r') as f:
                    source = f.read()
                print(f"📖 Loading {filename}...")
                result = self.execute(source, filename)
                if result is not None:
                    print(f"=> {result}")
            except Exception as e:
                print(f"❌ Error loading {filename}: {e}")
            return True
        elif cmd == "reset":
            self.interpreter = Interpreter(self.kid_mode, self.allow_files, self.allow_http)
            print("🧹 Interpreter reset.")
            return True
        elif cmd == "multiline" or cmd == "ml":
            self.in_multiline = True
            self.multiline_buffer = ""
            print("📝 Entering multi-line mode. Type 'end' on its own line to execute.")
            return True
        
        print(f"❓ Unknown command: {cmd}. Type '.help' for available commands.")
        return False
    
    def execute(self, source: str, filename: str = "<repl>"):
        try:
            lexer = Lexer(source, filename)
            tokens = lexer.scan_tokens()
            
            if lexer.has_errors():
                for error in lexer.errors:
                    print(error.show_with_context())
                return None
            
            parser = Parser(tokens, source)
            ast = parser.parse()
            
            if parser.has_errors():
                for error in parser.errors:
                    print(error.show_with_context())
                return None
            
            result = self.interpreter.interpret(ast, filename)
            return result
            
        except GlowError as e:
            print(e.show_with_context())
            return None
        except Exception as e:
            print(f"🔥 Internal error: {e}")
            if __debug__:
                traceback.print_exc()
            return None
    
    def show_help(self):
        print("GLOW Commands:")
        print("  .exit, .quit    - Exit the REPL")
        print("  .help          - Show this help")
        print("  .clear         - Clear screen and history")
        print("  .history       - Show command history")
        print("  .load <file>   - Load and run a GLOW file")
        print("  .reset         - Reset interpreter state")
        print("  .multiline     - Enter multi-line mode")
        print()
        print("GLOW 2.1 New Features:")
        print("  String interpolation: \"Hello {name}!\"")
        print("  Multi-line strings: \"\"\"...\"\"\"")
        print("  Array slicing: items[1:4] or items[::2]")
        print("  Negative indices: items[-1] (last item)")
        print("  'in' operator: if 'apple' in fruits")
        print("  Object property setting: set player.health to 50")
        print("  List comprehensions: [map numbers with n action n * 2]")
        print("  HTTP requests: do fetch \"https://api.example.com\"")
        print()
        print("Examples:")
        print("  say \"Hello {name}!\"")
        print("  set colors[1:3] to [\"red\", \"blue\"]")
        print("  if \"apple\" in fruits ... end")
        print("  set response to do fetch_json \"https://api.github.com/users/octocat\"")
        print("  set squares to [map numbers with n action n * n]")
    
    def show_history(self):
        if not self.history:
            print("No history yet.")
        else:
            for i, cmd in enumerate(self.history[-20:], 1):
                print(f"{i:3}: {cmd}")


# ============================================================================
# 11. FILE EXECUTION
# ============================================================================

def run_file(filename: str, kid_mode: bool = False, allow_files: bool = False, allow_http: bool = False):
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            source = f.read()
        
        lexer = Lexer(source, filename)
        tokens = lexer.scan_tokens()
        
        if lexer.has_errors():
            print("🌋 Lexer errors:")
            for error in lexer.errors:
                print(error.show_with_context())
            sys.exit(1)
        
        parser = Parser(tokens, source)
        ast = parser.parse()
        
        if parser.has_errors():
            print("🌋 Parser errors:")
            for error in parser.errors:
                print(error.show_with_context())
            sys.exit(1)
        
        interpreter = Interpreter(kid_mode, allow_files, allow_http)
        result = interpreter.interpret(ast, filename)
        
        if result is not None:
            print(f"=> {result}")
            
    except FileNotFoundError:
        print(f"❌ File not found: {filename}")
        sys.exit(1)
    except GlowError as e:
        print(e.show_with_context())
        sys.exit(1)
    except Exception as e:
        print(f"🔥 Internal error: {e}")
        sys.exit(1)


# ============================================================================
# 12. COMMAND LINE INTERFACE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="GLOW - A kid-friendly programming language",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  glow                         # Start REPL
  glow game.glo                # Run a GLOW file
  glow -e "say 'Hello!'"       # Execute code directly
  glow --kids game.glo         # Kid-friendly mode
  glow --allow-http            # Enable HTTP requests
        
        """
    )
    
    parser.add_argument(
        "file",
        nargs="?",
        help="GLOW script to run (.glo file)"
    )
    
    parser.add_argument(
        "-v", "--version",
        action="version",
        version="GLOW 2.1.0 - Phase 2 features"
    )
    
    parser.add_argument(
        "-e", "--execute",
        help="Execute code from command line"
    )
    
    parser.add_argument(
        "-k", "--kids",
        action="store_true",
        help="Enable kid-friendly mode"
    )
    
    parser.add_argument(
        "-s", "--safe",
        action="store_true",
        help="Enable safe mode (lower loop limits, more warnings)"
    )
    
    parser.add_argument(
        "-a", "--allow-files",
        action="store_true",
        help="Allow file operations"
    )
    
    parser.add_argument(
        "--allow-http",
        action="store_true",
        help="Allow HTTP requests (fetch)"
    )
    
    parser.add_argument(
        "-t", "--test",
        action="store_true",
        help="Run test suite"
    )
    
    args = parser.parse_args()
    
    kid_mode = args.kids or args.safe
    allow_files = args.allow_files or args.safe
    allow_http = args.allow_http
    
    if args.test:
        run_tests()
        return
    
    if args.execute:
        repl = GlowREPL(kid_mode, allow_files, allow_http)
        repl.execute(args.execute, "<command-line>")
    elif args.file:
        if not args.file.endswith('.glo'):
            print(f"⚠️  Warning: File doesn't have .glo extension")
        run_file(args.file, kid_mode, allow_files, allow_http)
    else:
        repl = GlowREPL(kid_mode, allow_files, allow_http)
        repl.run()


# ============================================================================
# 13. TEST SUITE (FIXED)
# ============================================================================

import unittest

class TestGLOW(unittest.TestCase):
    def test_string_methods_fixed(self):
        interpreter = Interpreter()
        
        # Test string method call with parentheses
        result = interpreter.interpret(Parser(
            Lexer('set result to "hello".uppercase()').scan_tokens()
        ).parse())
        
        result_val = interpreter.environment.get("result")
        self.assertEqual(result_val, "HELLO")
        
        # Test method without parentheses (returns bound method)
        result = interpreter.interpret(Parser(
            Lexer('set method to "hello".uppercase').scan_tokens()
        ).parse())
        
        method = interpreter.environment.get("method")
        self.assertIsInstance(method, BoundMethod)
        self.assertEqual(method(), "HELLO")
    
    def test_array_slicing(self):
        interpreter = Interpreter()
        
        # Test array slicing
        result = interpreter.interpret(Parser(
            Lexer('set items to [1, 2, 3, 4, 5]\nset slice to items[1:4]').scan_tokens()
        ).parse())
        
        items = interpreter.environment.get("items")
        slice_val = interpreter.environment.get("slice")
        self.assertEqual(slice_val, [2, 3, 4])
        
        # Test negative indices
        result = interpreter.interpret(Parser(
            Lexer('set last to items[-1]\nset last_two to items[-2:]').scan_tokens()
        ).parse())
        
        last = interpreter.environment.get("last")
        last_two = interpreter.environment.get("last_two")
        self.assertEqual(last, 5)
        self.assertEqual(last_two, [4, 5])
    
    def test_object_property_assignment(self):
        interpreter = Interpreter()
        
        # Test object property assignment
        result = interpreter.interpret(Parser(
            Lexer('set player to {health: 100}\nset player.health to 50\nset new_health to player.health').scan_tokens()
        ).parse())
        
        player = interpreter.environment.get("player")
        new_health = interpreter.environment.get("new_health")
        self.assertEqual(player["health"], 50)
        self.assertEqual(new_health, 50)
    
    def test_in_operator(self):
        interpreter = Interpreter()
        
        # Test 'in' operator
        result = interpreter.interpret(Parser(
            Lexer('set fruits to ["apple", "banana"]\nset has_apple to "apple" in fruits').scan_tokens()
        ).parse())
        
        has_apple = interpreter.environment.get("has_apple")
        self.assertTrue(has_apple)
        
        result = interpreter.interpret(Parser(
            Lexer('set text to "hello world"\nset has_world to "world" in text').scan_tokens()
        ).parse())
        
        has_world = interpreter.environment.get("has_world")
        self.assertTrue(has_world)
    
    def test_string_interpolation(self):
        interpreter = Interpreter()
        
        # Test string interpolation
        interpreter.environment.define("name", "Alice")
        interpreter.environment.define("age", 10)
        
        result = interpreter.interpret(Parser(
            Lexer('set greeting to "Hello {name}, you are {age} years old!"').scan_tokens()
        ).parse())
        
        greeting = interpreter.environment.get("greeting")
        self.assertEqual(greeting, "Hello Alice, you are 10 years old!")
    
    def test_list_comprehensions(self):
        interpreter = Interpreter()
        
        # Test list comprehension
        result = interpreter.interpret(Parser(
            Lexer('set numbers to [1, 2, 3, 4, 5]\nset doubles to [map numbers with n action n * 2]').scan_tokens()
        ).parse())
        
        doubles = interpreter.environment.get("doubles")
        self.assertEqual(doubles, [2, 4, 6, 8, 10])
        
        # Test with filter
        result = interpreter.interpret(Parser(
            Lexer('set evens to [map numbers with n where n % 2 is 0 action n]').scan_tokens()
        ).parse())
        
        evens = interpreter.environment.get("evens")
        self.assertEqual(evens, [2, 4])
    
    def test_for_each_with_index_fixed(self):
        interpreter = Interpreter()
        
        # Test for each with index - using to_string() instead of str()
        result = interpreter.interpret(Parser(
            Lexer('set items to ["a", "b", "c"]\nset result to ""\nfor each item, idx in items\nset result to result + item + do to_string idx\nend').scan_tokens()
        ).parse())
        
        result_val = interpreter.environment.get("result")
        self.assertEqual(result_val, "a0b1c2")


def run_tests():
    print("🧪 Running GLOW test suite...")
    suite = unittest.TestLoader().loadTestsFromTestCase(TestGLOW)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    if result.wasSuccessful():
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed.")
        sys.exit(1)


# ============================================================================
# 14. EXAMPLE PROGRAMS
# ============================================================================

EXAMPLE_PROGRAM = """// GLOW 2.1 - Complete Example
say "🌟 Welcome to GLOW 2.1! 🌟"

// 1. String interpolation
set name to "Alex"
set age to 12
say "Hello {name}, you are {age} years old!"

// 2. Multi-line strings
set poem to """ '''
Roses are red
Violets are blue
GLOW is awesome
And so are you! '''
"""
say poem

// 3. Array operations
set numbers to [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

// Slicing
say "First three: {numbers[0:3]}"
say "Last three: {numbers[-3:]}"
say "Every other: {numbers[::2]}"

// Negative indices
say "Last item: {numbers[-1]}"
say "Second last: {numbers[-2]}"

// 4. 'in' operator
if 30 in numbers
    say "✓ Found 30 in the list!"
end

if "blue" in poem
    say "✓ 'blue' is in the poem!"
end

// 5. Object property assignment
set player to {
    name: "Hero",
    health: 100,
    inventory: ["sword", "potion"]
}

set player.health to 75
set player.inventory[1] to "super potion"
say "Player {player.name} has {player.health} health and {player.inventory}"

// 6. List comprehensions
set numbers to [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

// Double all numbers
set doubled to [map numbers with n action n * 2]
say "Doubled: {doubled}"

// Only even numbers
set evens to [map numbers with n where n % 2 is 0 action n]
say "Evens: {evens}"

// Square numbers greater than 5
set big_squares to [map numbers with n where n > 5 action n * n]
say "Squares of numbers > 5: {big_squares}"

// 7. String methods (FIXED!)
set message to "  hello GLOW world!  "
say "Original: '{message}'"
say "Trimmed: '{message.trim()}'"
say "Uppercase: {message.trim().uppercase()}"
say "Contains 'GLOW': {message.contains('GLOW')}"
say "Replaced: {message.replace('world', 'programmer')}"

// 8. Try/rescue with string methods
try
    set result to "hello".bad_method()  // This will fail
    say "Result: {result}"
rescue as error
    say "✓ Safely caught error: {error}"
end

// 9. HTTP requests (if allowed)
/*
set response to do fetch "https://api.github.com/users/octocat"
if response.ok
    say "GitHub user: {response.body.login}"
    say "Followers: {response.body.followers}"
else
    say "Error: {response.error}"
end
*/

// 10. Complex example: Quiz game
say ""
say "🧠 QUIZ TIME! 🧠"

set score to 0
set questions to [
    {
        question: "What is 5 + 3?",
        answer: "8",
        options: ["6", "7", "8", "9"]
    },
    {
        question: "What is the capital of France?",
        answer: "Paris",
        options: ["London", "Berlin", "Paris", "Madrid"]
    },
    {
        question: "Which animal says 'meow'?",
        answer: "Cat",
        options: ["Dog", "Cat", "Cow", "Bird"]
    }
]

for each q, i in questions
    say ""
    say "Question #{i + 1}: {q.question}"
    
    for each option, j in q.options
        say "  {j + 1}. {option}"
    end
    
    try
        ask "Your answer (1-{q.options.length}): " save as answer_str
        set answer_num to do to_number answer_str
        set chosen to q.options[answer_num - 1]
        
        if chosen is q.answer
            say "✅ Correct!"
            set score to score + 1
        else
            say "❌ The answer is {q.answer}"
        end
    rescue as error
        say "⚠️  That wasn't a valid answer: {error}"
    end
end

say ""
say "📊 FINAL SCORE: {score} out of {questions.length}"

if score is questions.length
    say "🎉 PERFECT SCORE! You're a genius! 🎉"
else if score > 0
    say "👍 Good job!"
else
    say "💪 Try again!"
end

say ""
say "✨ GLOW 2.1 Demo Complete! ✨"
"""


# ============================================================================
# 15. MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # Create examples directory
    examples_dir = Path("examples")
    examples_dir.mkdir(exist_ok=True)
    
    # Write example files with UTF-8 encoding
    (examples_dir / "example.glo").write_text(EXAMPLE_PROGRAM, encoding='utf-8')
    
    # Quick demo if no arguments
    if len(sys.argv) == 1:
        print("=" * 60)
        print("🌟 GLOW 2.1 - Phase 2 Features 🌟")
        print("=" * 60)
        print()
        print("All Critical Issues Fixed:")
        print("  ✅ String methods work: 'hello'.uppercase()")
        print("  ✅ Array slicing: items[1:4] or items[-1]")
        print("  ✅ Object properties: set player.health to 50")
        print("  ✅ 'in' operator: if 'apple' in fruits")
        print("  ✅ String interpolation: \"Hello {name}!\"")
        print("  ✅ List comprehensions: [map items with x action x*2]")
        print("  ✅ HTTP requests: do fetch \"https://...\"")
        print()
        print("Try: python glow.py --kids examples/example.glo")
        print("Or: python glow.py --allow-http (for HTTP features)")
        print("Or: python glow.py (for REPL)")
        print("Or: python glow.py -t (run tests)")
        print()
        print("-" * 40)
        print("Quick demo:")
        
        # Quick working demo
        interpreter = Interpreter(kid_mode=True)
        demo_code = '''
say "✨ GLOW 2.1 Demo ✨"

// String interpolation works!
set name to "World"
say "Hello {name}!"

// String methods work!
say "hello".uppercase() + " " + "WORLD".lowercase()

// Array slicing works!
set nums to [1, 2, 3, 4, 5]
say "First 3: {nums[0:3]}"
say "Last: {nums[-1]}"

// 'in' operator works!
if 3 in nums
    say "✓ Found 3 in the list!"
end

// List comprehension works!
set squares to [map nums with n action n * n]
say "Squares: {squares}"

say "✅ Everything works!"
'''
        
        lexer = Lexer(demo_code, "<demo>")
        tokens = lexer.scan_tokens()
        
        if not lexer.has_errors():
            parser = Parser(tokens, demo_code)
            ast = parser.parse()
            
            if not parser.has_errors():
                interpreter.interpret(ast)
            else:
                print("Parser errors in demo!")
        else:
            print("Lexer errors in demo!")
    else:
        main()
