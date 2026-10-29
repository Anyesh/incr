use crate::coord::Coord;
use crate::value::{CellValue, DIV0, PARSE, VALUE};

/// The most cells one range may name, so a typo such as `A1:A9999999`
/// fails to parse instead of allocating millions of refs.
const MAX_RANGE_CELLS: u64 = 1 << 20;

const FUNCTIONS: [&str; 5] = ["SUM", "AVG", "MIN", "MAX", "IF"];

#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    Number(f64),
    CellRef(Coord),
    BinOp(Box<Expr>, Op, Box<Expr>),
    UnaryNeg(Box<Expr>),
    FnCall(String, Vec<Expr>),
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Op {
    Add,
    Sub,
    Mul,
    Div,
    Gt,
    Lt,
    Gte,
    Lte,
    Eq,
}

// Tokenizer

#[derive(Debug, Clone, PartialEq)]
enum Token {
    Number(f64),
    Ident(String),
    Plus,
    Minus,
    Star,
    Slash,
    LParen,
    RParen,
    Comma,
    Colon,
    Gt,
    Lt,
    Gte,
    Lte,
    EqEq,
}

fn tokenize(input: &str) -> Result<Vec<Token>, ()> {
    let mut tokens = Vec::new();
    let chars: Vec<char> = input.chars().collect();
    let mut i = 0;

    while i < chars.len() {
        match chars[i] {
            ' ' | '\t' => {
                i += 1;
            }
            '+' => {
                tokens.push(Token::Plus);
                i += 1;
            }
            '*' => {
                tokens.push(Token::Star);
                i += 1;
            }
            '/' => {
                tokens.push(Token::Slash);
                i += 1;
            }
            '(' => {
                tokens.push(Token::LParen);
                i += 1;
            }
            ')' => {
                tokens.push(Token::RParen);
                i += 1;
            }
            ',' => {
                tokens.push(Token::Comma);
                i += 1;
            }
            ':' => {
                tokens.push(Token::Colon);
                i += 1;
            }
            '-' => {
                tokens.push(Token::Minus);
                i += 1;
            }
            '>' => {
                if i + 1 < chars.len() && chars[i + 1] == '=' {
                    tokens.push(Token::Gte);
                    i += 2;
                } else {
                    tokens.push(Token::Gt);
                    i += 1;
                }
            }
            '<' => {
                if i + 1 < chars.len() && chars[i + 1] == '=' {
                    tokens.push(Token::Lte);
                    i += 2;
                } else {
                    tokens.push(Token::Lt);
                    i += 1;
                }
            }
            '=' => {
                if i + 1 < chars.len() && chars[i + 1] == '=' {
                    tokens.push(Token::EqEq);
                    i += 2;
                } else {
                    tokens.push(Token::EqEq);
                    i += 1;
                }
            }
            c if c.is_ascii_digit() || c == '.' => {
                let start = i;
                while i < chars.len() && (chars[i].is_ascii_digit() || chars[i] == '.') {
                    i += 1;
                }
                let s: String = chars[start..i].iter().collect();
                let n = s.parse::<f64>().map_err(|_| ())?;
                tokens.push(Token::Number(n));
            }
            c if c.is_ascii_alphabetic() || c == '_' => {
                let start = i;
                while i < chars.len() && (chars[i].is_ascii_alphanumeric() || chars[i] == '_') {
                    i += 1;
                }
                let s: String = chars[start..i].iter().collect();
                tokens.push(Token::Ident(s));
            }
            _ => return Err(()),
        }
    }

    Ok(tokens)
}

// Recursive descent parser

struct Parser {
    tokens: Vec<Token>,
    pos: usize,
}

impl Parser {
    fn new(tokens: Vec<Token>) -> Self {
        Self { tokens, pos: 0 }
    }

    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.pos)
    }

    fn advance(&mut self) -> Option<Token> {
        if self.pos < self.tokens.len() {
            let t = self.tokens[self.pos].clone();
            self.pos += 1;
            Some(t)
        } else {
            None
        }
    }

    fn expect(&mut self, expected: &Token) -> Result<(), ()> {
        if self.peek() == Some(expected) {
            self.advance();
            Ok(())
        } else {
            Err(())
        }
    }

    fn parse_expr(&mut self) -> Result<Expr, ()> {
        self.parse_comparison()
    }

    fn parse_comparison(&mut self) -> Result<Expr, ()> {
        let mut left = self.parse_additive()?;

        loop {
            let op = match self.peek() {
                Some(Token::Gt) => Op::Gt,
                Some(Token::Lt) => Op::Lt,
                Some(Token::Gte) => Op::Gte,
                Some(Token::Lte) => Op::Lte,
                Some(Token::EqEq) => Op::Eq,
                _ => break,
            };
            self.advance();
            let right = self.parse_additive()?;
            left = Expr::BinOp(Box::new(left), op, Box::new(right));
        }

        Ok(left)
    }

    fn parse_additive(&mut self) -> Result<Expr, ()> {
        let mut left = self.parse_multiplicative()?;

        loop {
            let op = match self.peek() {
                Some(Token::Plus) => Op::Add,
                Some(Token::Minus) => Op::Sub,
                _ => break,
            };
            self.advance();
            let right = self.parse_multiplicative()?;
            left = Expr::BinOp(Box::new(left), op, Box::new(right));
        }

        Ok(left)
    }

    fn parse_multiplicative(&mut self) -> Result<Expr, ()> {
        let mut left = self.parse_unary()?;

        loop {
            let op = match self.peek() {
                Some(Token::Star) => Op::Mul,
                Some(Token::Slash) => Op::Div,
                _ => break,
            };
            self.advance();
            let right = self.parse_unary()?;
            left = Expr::BinOp(Box::new(left), op, Box::new(right));
        }

        Ok(left)
    }

    fn parse_unary(&mut self) -> Result<Expr, ()> {
        if self.peek() == Some(&Token::Minus) {
            self.advance();
            let expr = self.parse_unary()?;
            return Ok(Expr::UnaryNeg(Box::new(expr)));
        }
        self.parse_primary()
    }

    fn parse_primary(&mut self) -> Result<Expr, ()> {
        match self.peek().cloned() {
            Some(Token::Number(n)) => {
                self.advance();
                Ok(Expr::Number(n))
            }
            Some(Token::Ident(name)) => {
                self.advance();
                if self.peek() == Some(&Token::LParen) {
                    self.advance();
                    let name = name.to_uppercase();
                    if !FUNCTIONS.contains(&name.as_str()) {
                        return Err(());
                    }
                    let args = self.parse_fn_args()?;
                    self.expect(&Token::RParen)?;
                    if name == "IF" && args.len() != 3 {
                        return Err(());
                    }
                    Ok(Expr::FnCall(name, args))
                } else {
                    Coord::parse(&name).map(Expr::CellRef).ok_or(())
                }
            }
            Some(Token::LParen) => {
                self.advance();
                let expr = self.parse_expr()?;
                self.expect(&Token::RParen)?;
                Ok(expr)
            }
            _ => Err(()),
        }
    }

    /// Parse function arguments, expanding range expressions like A1:B5.
    fn parse_fn_args(&mut self) -> Result<Vec<Expr>, ()> {
        let mut args = Vec::new();
        if self.peek() == Some(&Token::RParen) {
            return Ok(args);
        }

        loop {
            // Try to parse a range: IDENT:IDENT
            let first = self.parse_expr()?;

            if self.peek() == Some(&Token::Colon) {
                self.advance();
                let second = self.parse_expr()?;
                // Both must be cell refs for a range
                if let (Expr::CellRef(start), Expr::CellRef(end)) = (first, second) {
                    args.extend(expand_range(start, end)?.into_iter().map(Expr::CellRef));
                } else {
                    return Err(());
                }
            } else {
                args.push(first);
            }

            if self.peek() == Some(&Token::Comma) {
                self.advance();
            } else {
                break;
            }
        }

        Ok(args)
    }
}

fn expand_range(start: Coord, end: Coord) -> Result<Vec<Coord>, ()> {
    let (c_lo, c_hi) = (start.col.min(end.col), start.col.max(end.col));
    let (r_lo, r_hi) = (start.row.min(end.row), start.row.max(end.row));
    let count = u64::from(c_hi - c_lo + 1) * u64::from(r_hi - r_lo + 1);
    if count > MAX_RANGE_CELLS {
        return Err(());
    }
    Ok((r_lo..=r_hi)
        .flat_map(|r| (c_lo..=c_hi).map(move |c| Coord::new(c, r)))
        .collect())
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ParseError;

pub fn parse_formula(input: &str) -> Result<Expr, ParseError> {
    let tokens = tokenize(input).map_err(|()| ParseError)?;
    let mut parser = Parser::new(tokens);
    let expr = parser.parse_expr().map_err(|()| ParseError)?;
    if parser.pos != parser.tokens.len() {
        return Err(ParseError);
    }
    Ok(expr)
}

/// What a cell's raw content means.
#[derive(Debug, Clone, PartialEq)]
pub enum Content {
    Empty,
    Number(f64),
    Text(String),
    Formula(Expr),
    /// A `=` content that does not parse.
    Invalid,
}

pub fn parse_content(raw: &str) -> Content {
    if raw.is_empty() {
        Content::Empty
    } else if let Some(formula) = raw.strip_prefix('=') {
        parse_formula(formula).map_or(Content::Invalid, Content::Formula)
    } else if let Ok(n) = raw.trim().parse::<f64>() {
        Content::Number(n)
    } else {
        Content::Text(raw.to_string())
    }
}

impl Content {
    /// The cell's value, reading other cells through `lookup`.
    pub fn evaluate(&self, lookup: &mut impl FnMut(Coord) -> CellValue) -> CellValue {
        match self {
            Content::Empty => CellValue::Empty,
            Content::Number(n) => CellValue::Number(*n),
            Content::Text(s) => CellValue::Text(s.clone()),
            Content::Formula(expr) => eval_expr(expr, lookup),
            Content::Invalid => CellValue::Error(PARSE),
        }
    }

    /// Every cell the formula names, both `IF` branches included.
    pub fn precedents(&self) -> Vec<Coord> {
        let mut out = Vec::new();
        if let Content::Formula(expr) = self {
            expr.collect_refs(&mut out);
        }
        out
    }
}

impl Expr {
    fn collect_refs(&self, out: &mut Vec<Coord>) {
        match self {
            Expr::Number(_) => {}
            Expr::CellRef(at) => out.push(*at),
            Expr::BinOp(lhs, _, rhs) => {
                lhs.collect_refs(out);
                rhs.collect_refs(out);
            }
            Expr::UnaryNeg(inner) => inner.collect_refs(out),
            Expr::FnCall(_, args) => args.iter().for_each(|a| a.collect_refs(out)),
        }
    }
}

/// Evaluates `expr`, reading cells through `lookup`. Both operands of an
/// operator and every argument of an aggregate are read before an error
/// is returned, so the cells read depend only on `IF` conditions; an
/// engine that records reads as dependencies then sees the same cycles
/// as one that reads formulas statically, `IF` aside.
pub fn eval_expr(expr: &Expr, lookup: &mut impl FnMut(Coord) -> CellValue) -> CellValue {
    eval(expr, lookup).unwrap_or_else(CellValue::Error)
}

type Eval = Result<CellValue, &'static str>;

fn eval(expr: &Expr, lookup: &mut impl FnMut(Coord) -> CellValue) -> Eval {
    match expr {
        Expr::Number(n) => Ok(CellValue::Number(*n)),
        Expr::CellRef(at) => match lookup(*at) {
            CellValue::Empty => Ok(CellValue::Number(0.0)),
            CellValue::Error(code) => Err(code),
            v => Ok(v),
        },
        Expr::UnaryNeg(inner) => Ok(CellValue::Number(-number(eval(inner, lookup))?)),
        Expr::BinOp(lhs, op, rhs) => {
            let l = eval(lhs, lookup);
            let r = eval(rhs, lookup);
            let (l, r) = (number(l)?, number(r)?);
            binop(l, *op, r).map(CellValue::Number)
        }
        Expr::FnCall(name, args) if name == "IF" => {
            let cond = number(eval(&args[0], lookup))?;
            let taken = if cond != 0.0 && !cond.is_nan() { 1 } else { 2 };
            eval(&args[taken], lookup)
        }
        Expr::FnCall(name, args) => aggregate(name, args, lookup),
    }
}

fn number(v: Eval) -> Result<f64, &'static str> {
    match v? {
        CellValue::Number(n) => Ok(n),
        CellValue::Empty => Ok(0.0),
        CellValue::Text(_) => Err(VALUE),
        CellValue::Error(code) => Err(code),
    }
}

fn binop(l: f64, op: Op, r: f64) -> Result<f64, &'static str> {
    let truth = |b: bool| if b { 1.0 } else { 0.0 };
    Ok(match op {
        Op::Add => l + r,
        Op::Sub => l - r,
        Op::Mul => l * r,
        Op::Div if r == 0.0 => return Err(DIV0),
        Op::Div => l / r,
        Op::Gt => truth(l > r),
        Op::Lt => truth(l < r),
        Op::Gte => truth(l >= r),
        Op::Lte => truth(l <= r),
        Op::Eq => truth((l - r).abs() < f64::EPSILON),
    })
}

/// SUM, AVG, MIN and MAX over their arguments' numbers. A referenced
/// cell holding text or nothing is skipped; a computed argument must be
/// a number.
fn aggregate(name: &str, args: &[Expr], lookup: &mut impl FnMut(Coord) -> CellValue) -> Eval {
    let values: Vec<Eval> = args
        .iter()
        .map(|arg| match arg {
            Expr::CellRef(at) => Ok(lookup(*at)),
            _ => eval(arg, lookup).and_then(|v| match v {
                CellValue::Text(_) => Err(VALUE),
                v => Ok(v),
            }),
        })
        .collect();
    let mut numbers = Vec::with_capacity(values.len());
    for v in values {
        match v? {
            CellValue::Number(n) => numbers.push(n),
            CellValue::Error(code) => return Err(code),
            CellValue::Empty | CellValue::Text(_) => {}
        }
    }
    let result = match name {
        "SUM" => numbers.iter().sum(),
        "AVG" if numbers.is_empty() => return Err(DIV0),
        "AVG" => numbers.iter().sum::<f64>() / numbers.len() as f64,
        "MIN" if numbers.is_empty() => 0.0,
        "MIN" => numbers.iter().copied().fold(f64::INFINITY, f64::min),
        "MAX" if numbers.is_empty() => 0.0,
        "MAX" => numbers.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        _ => unreachable!("the parser admits only FUNCTIONS"),
    };
    Ok(CellValue::Number(result))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_number() {
        let expr = parse_formula("42").unwrap();
        assert!(matches!(expr, Expr::Number(n) if (n - 42.0).abs() < f64::EPSILON));
    }

    #[test]
    fn parse_cell_ref_expr() {
        let expr = parse_formula("A1").unwrap();
        assert_eq!(expr, Expr::CellRef(Coord::new(0, 1)));
    }

    #[test]
    fn parse_binary() {
        let expr = parse_formula("A1+B2*3").unwrap();
        // Should parse as A1 + (B2 * 3) due to precedence
        assert!(matches!(expr, Expr::BinOp(_, Op::Add, _)));
    }

    #[test]
    fn parse_parens() {
        let expr = parse_formula("(A1+B1)*2").unwrap();
        assert!(matches!(expr, Expr::BinOp(_, Op::Mul, _)));
    }

    #[test]
    fn parse_function() {
        let expr = parse_formula("SUM(A1:A3)").unwrap();
        if let Expr::FnCall(name, args) = expr {
            assert_eq!(name, "SUM");
            assert_eq!(args.len(), 3); // A1, A2, A3
        } else {
            panic!("expected FnCall");
        }
    }

    #[test]
    fn parse_negative() {
        let expr = parse_formula("-5").unwrap();
        assert!(matches!(expr, Expr::UnaryNeg(_)));
    }

    #[test]
    fn expand_range_rect() {
        let at = |name| Coord::parse(name).unwrap();
        let cells = expand_range(at("B3"), at("A1")).unwrap();
        let names: Vec<String> = cells.iter().map(Coord::to_string).collect();
        assert_eq!(names, vec!["A1", "B1", "A2", "B2", "A3", "B3"]);
    }
}
