//! The Local instances of incr-python-persist's overlay journal.

use crate::persist::LocalBinding;

pub(crate) use incr_python_persist::overlay::Born;

pub(crate) type Undo = incr_python_persist::overlay::Undo<LocalBinding>;
