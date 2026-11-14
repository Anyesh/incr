use std::any::Any;
use std::cell::RefCell;
use std::fmt;
use std::sync::Arc;

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

/// A binding's own state for its value code, such as a table of codecs,
/// that a [`Store`](crate::Store) makes current through
/// [`current_value_scope`] while it holds its lock, on whichever thread
/// takes it. Set it in [`Options::value_scope`](crate::Options::value_scope).
#[derive(Clone)]
pub struct ValueScope(Arc<dyn Any + Send + Sync>);

impl ValueScope {
    pub fn new<T: Any + Send + Sync>(state: Arc<T>) -> Self {
        ValueScope(state)
    }

    /// The state this scope was created with, if it is a `T`.
    pub fn downcast<T: Any + Send + Sync>(&self) -> Option<Arc<T>> {
        Arc::clone(&self.0).downcast().ok()
    }
}

impl fmt::Debug for ValueScope {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("ValueScope(..)")
    }
}

thread_local! {
    static SCOPE: RefCell<Option<ValueScope>> = const { RefCell::new(None) };
}

/// The value scope entered on this thread, if any.
pub fn current_value_scope() -> Option<ValueScope> {
    SCOPE.with(|scope| scope.borrow().clone())
}

/// Makes `scope` current on this thread until dropped, then restores the
/// previous one, also when unwinding.
pub fn enter_value_scope(scope: ValueScope) -> ValueScopeGuard {
    let previous = SCOPE.with(|current| current.replace(Some(scope)));
    ValueScopeGuard { previous }
}

#[must_use = "the scope ends when the guard is dropped"]
pub struct ValueScopeGuard {
    previous: Option<ValueScope>,
}

impl Drop for ValueScopeGuard {
    fn drop(&mut self) {
        let previous = self.previous.take();
        let _ = SCOPE.try_with(|current| current.replace(previous));
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
