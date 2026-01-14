// Core types and traits to and from python

use diffsol::Scalar;
use numpy::{Element, PyUntypedArray};
use pyo3::PyAny;

// Trait for valid matrix and python scalar types (f32 and f64 currently used)
pub(crate) trait PyCompatibleScalar: Scalar + Element {}
impl<T> PyCompatibleScalar for T where T: Scalar + Element {}

// Solve methods would ideally use something like a dimensioned PyUntypedArray in their return
// types, but this has two problems: 1) it cannot be dimensioned, 2) it cannot be returned (it
// does not have a .into method). So instead this code falls back to PyAny using these aliases
// to communicate what a paricular return type _should_ be.
pub(crate) type PyUntypedArray1 = PyAny;
pub(crate) type PyUntypedArray2 = PyAny;

// Similarly, there are no dimensioned read only untyped arrays, so aliases are used as above.
pub(crate) type PyReadonlyUntypedArray2 = PyUntypedArray;
