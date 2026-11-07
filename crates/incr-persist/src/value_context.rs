use std::cell::RefCell;

/// Which slot a user value is being encoded or decoded for, visible to its
/// `Serialize` and `Deserialize` impls through [`current_value_context`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ValueContext {
    /// An input or collection of a [`Store`](crate::Store), by its key.
    Key(String),
    /// The key or the value of a memo table's entry.
    MemoTable { table: String, part: MemoPart },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemoPart {
    Key,
    Value,
}

/// A memo entry the last checkpoints left out because its key or value did
/// not encode, or did not decode back equal to itself. The entry is
/// recomputed after the next restart; nothing wrong is stored. At most one
/// per table and part is kept.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EncodeMismatch {
    pub table: String,
    pub part: MemoPart,
}

thread_local! {
    static CURRENT: RefCell<Option<ValueContext>> = const { RefCell::new(None) };
}

/// The context of the encode or decode running on this thread, if any.
pub fn current_value_context() -> Option<ValueContext> {
    CURRENT.with(|current| current.borrow().clone())
}

/// Makes `context` current on this thread until dropped, then restores the
/// previous one, also when unwinding.
pub fn enter_value_context(context: ValueContext) -> ValueContextGuard {
    let previous = CURRENT.with(|current| current.replace(Some(context)));
    ValueContextGuard { previous }
}

#[must_use = "the context ends when the guard is dropped"]
pub struct ValueContextGuard {
    previous: Option<ValueContext>,
}

impl Drop for ValueContextGuard {
    fn drop(&mut self) {
        let previous = self.previous.take();
        let _ = CURRENT.try_with(|current| current.replace(previous));
    }
}

pub(crate) fn in_key<R>(key: &str, f: impl FnOnce() -> R) -> R {
    let _guard = enter_value_context(ValueContext::Key(key.to_owned()));
    f()
}

pub(crate) fn in_memo<R>(table: &str, part: MemoPart, f: impl FnOnce() -> R) -> R {
    let _guard = enter_value_context(ValueContext::MemoTable {
        table: table.to_owned(),
        part,
    });
    f()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(name: &str) -> ValueContext {
        ValueContext::Key(name.to_owned())
    }

    #[test]
    fn no_context_outside_a_scope() {
        assert_eq!(current_value_context(), None);
    }

    #[test]
    fn nested_guards_restore_in_order() {
        let outer = enter_value_context(key("a"));
        {
            let _inner = enter_value_context(key("b"));
            assert_eq!(current_value_context(), Some(key("b")));
        }
        assert_eq!(current_value_context(), Some(key("a")));
        drop(outer);
        assert_eq!(current_value_context(), None);
    }

    #[test]
    fn a_panic_restores_the_previous_context() {
        let _outer = enter_value_context(key("a"));
        let result = std::panic::catch_unwind(|| {
            let _inner = enter_value_context(key("b"));
            panic!("unwind");
        });
        assert!(result.is_err());
        assert_eq!(current_value_context(), Some(key("a")));
    }

    #[test]
    fn contexts_are_per_thread() {
        let _guard = enter_value_context(key("a"));
        std::thread::spawn(|| assert_eq!(current_value_context(), None))
            .join()
            .unwrap();
    }
}
