use std::collections::HashMap;

use incr_concurrent::{Incr, Runtime};

// ---------------------------------------------------------------------------
// AST
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub enum Expr {
    Number(f64),
    CellRef(String),
    BinOp(Box<Expr>, Op, Box<Expr>),
    UnaryNeg(Box<Expr>),
    FnCall(String, Vec<Expr>),
}

#[derive(Debug, Clone, Copy)]
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

// ---------------------------------------------------------------------------
// Tokenizer
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Recursive descent parser
// ---------------------------------------------------------------------------

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
                // Check if this is a function call
                if self.peek() == Some(&Token::LParen) {
                    self.advance(); // consume '('
                    let args = self.parse_fn_args(&name)?;
                    self.expect(&Token::RParen)?;
                    Ok(Expr::FnCall(name.to_uppercase(), args))
                } else {
                    // Cell reference
                    Ok(Expr::CellRef(name.to_uppercase()))
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
    fn parse_fn_args(&mut self, _fn_name: &str) -> Result<Vec<Expr>, ()> {
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
                if let (Expr::CellRef(start), Expr::CellRef(end)) = (&first, &second) {
                    let expanded = expand_range(start, end)?;
                    for cell in expanded {
                        args.push(Expr::CellRef(cell));
                    }
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

// ---------------------------------------------------------------------------
// Range expansion
// ---------------------------------------------------------------------------

fn parse_cell_ref(s: &str) -> Result<(u8, u32), ()> {
    let s = s.to_uppercase();
    let col_char = s.chars().next().ok_or(())?;
    if !col_char.is_ascii_alphabetic() {
        return Err(());
    }
    let col = col_char as u8 - b'A';
    let row: u32 = s[1..].parse().map_err(|_| ())?;
    Ok((col, row))
}

fn cell_name(col: u8, row: u32) -> String {
    format!("{}{}", (b'A' + col) as char, row)
}

fn expand_range(start: &str, end: &str) -> Result<Vec<String>, ()> {
    let (sc, sr) = parse_cell_ref(start)?;
    let (ec, er) = parse_cell_ref(end)?;
    let c_lo = sc.min(ec);
    let c_hi = sc.max(ec);
    let r_lo = sr.min(er);
    let r_hi = sr.max(er);

    let mut cells = Vec::new();
    for r in r_lo..=r_hi {
        for c in c_lo..=c_hi {
            cells.push(cell_name(c, r));
        }
    }
    Ok(cells)
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

pub fn parse_formula(input: &str) -> Result<Expr, ()> {
    let tokens = tokenize(input)?;
    let mut parser = Parser::new(tokens);
    let expr = parser.parse_expr()?;
    if parser.pos != parser.tokens.len() {
        return Err(()); // unconsumed tokens
    }
    Ok(expr)
}

/// Evaluate an AST against the incr runtime, calling rt.get() on cell
/// value nodes to establish dynamic dependencies.
pub fn eval_expr(expr: &Expr, rt: &Runtime, value_nodes: &HashMap<String, Incr<f64>>) -> f64 {
    match expr {
        Expr::Number(n) => *n,
        Expr::CellRef(name) => {
            if let Some(&handle) = value_nodes.get(name) {
                rt.get(handle)
            } else {
                f64::NAN
            }
        }
        Expr::UnaryNeg(inner) => -eval_expr(inner, rt, value_nodes),
        Expr::BinOp(lhs, op, rhs) => {
            let l = eval_expr(lhs, rt, value_nodes);
            let r = eval_expr(rhs, rt, value_nodes);
            match op {
                Op::Add => l + r,
                Op::Sub => l - r,
                Op::Mul => l * r,
                Op::Div => {
                    if r == 0.0 {
                        f64::INFINITY
                    } else {
                        l / r
                    }
                }
                Op::Gt => {
                    if l > r {
                        1.0
                    } else {
                        0.0
                    }
                }
                Op::Lt => {
                    if l < r {
                        1.0
                    } else {
                        0.0
                    }
                }
                Op::Gte => {
                    if l >= r {
                        1.0
                    } else {
                        0.0
                    }
                }
                Op::Lte => {
                    if l <= r {
                        1.0
                    } else {
                        0.0
                    }
                }
                Op::Eq => {
                    if (l - r).abs() < f64::EPSILON {
                        1.0
                    } else {
                        0.0
                    }
                }
            }
        }
        Expr::FnCall(name, args) => match name.as_str() {
            "SUM" => args.iter().map(|a| eval_expr(a, rt, value_nodes)).sum(),
            "AVG" => {
                if args.is_empty() {
                    return f64::NAN;
                }
                let sum: f64 = args.iter().map(|a| eval_expr(a, rt, value_nodes)).sum();
                sum / args.len() as f64
            }
            "MIN" => args
                .iter()
                .map(|a| eval_expr(a, rt, value_nodes))
                .fold(f64::INFINITY, f64::min),
            "MAX" => args
                .iter()
                .map(|a| eval_expr(a, rt, value_nodes))
                .fold(f64::NEG_INFINITY, f64::max),
            "IF" => {
                if args.len() != 3 {
                    return f64::NAN;
                }
                let cond = eval_expr(&args[0], rt, value_nodes);
                if cond != 0.0 && !cond.is_nan() {
                    eval_expr(&args[1], rt, value_nodes)
                } else {
                    eval_expr(&args[2], rt, value_nodes)
                }
            }
            _ => f64::NAN,
        },
    }
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
        assert!(matches!(expr, Expr::CellRef(ref s) if s == "A1"));
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
        let cells = expand_range("A1", "B3").unwrap();
        assert_eq!(cells, vec!["A1", "B1", "A2", "B2", "A3", "B3"]);
    }
}
