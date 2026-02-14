use diffsol::{BdfState, DefaultDenseMatrix, OdeSolverState, RkState, StateRef, Vector, VectorCommon, VectorHost, MatrixCommon};
use pyo3::{Bound, Python};
use num_traits::FromPrimitive;

use crate::py_types::{PyUntypedArray1, PyUntypedArray2};
use crate::py_convert::{MatrixToPy, VectorToPy};



pub(crate) trait PySolution {
    fn get_ys<'py>(&self, py: Python<'py>) -> Bound<'py, PyUntypedArray2>;
    fn get_ts<'py>(&self, py: Python<'py>) -> Bound<'py, PyUntypedArray1>;
    fn get_sens<'py>(&self, py: Python<'py>) -> Vec<Bound<'py, PyUntypedArray2>>;
    fn set_state_y(&mut self, y: &[f64]);
    fn get_state_y<'py>(&self, py: Python<'py>) -> Bound<'py, PyUntypedArray1>;
}


pub(crate) enum GenericPyState<V: Vector + DefaultDenseMatrix> {
    Bdf(BdfState<V>),
    Rk(RkState<V>),
}

pub(crate) struct GenericPySolution<V: Vector + DefaultDenseMatrix> {
    state: GenericPyState<V>,
    ys: <V as DefaultDenseMatrix>::M,
    ts: Option<V>,
    sens: Vec<<V as DefaultDenseMatrix>::M>,
}

fn copy_slice_to_vec<V: VectorHost>(state: &mut V, y: &[f64]) {
    for (yi, &y_val) in state.as_mut_slice().iter_mut().zip(y.iter()) {
        *yi = V::T::from_f64(y_val).unwrap();
    }
}


impl<V: VectorHost + DefaultDenseMatrix> PySolution for GenericPySolution<V> 
where 
    for<'b> <V as VectorCommon>::Inner: VectorToPy<'b, V::T>,
    for<'b> <<V as DefaultDenseMatrix>::M as MatrixCommon>::Inner: MatrixToPy<'b, V::T>,

{
    fn set_state_y(&mut self, y: &[f64]) {
        match &mut self.state {
            GenericPyState::Bdf(state) => copy_slice_to_vec(state.as_mut().y, y),
            GenericPyState::Rk(state) => copy_slice_to_vec(state.as_mut().y, y),
        }
    }

    fn get_state_y<'py>(&self, py: Python<'py>) -> Bound<'py, PyUntypedArray1> {
        match &self.state {
            GenericPyState::Bdf(state) => state.as_ref().y.inner().to_pyarray1(py).into_any(),
            GenericPyState::Rk(state) => state.as_ref().y.inner().to_pyarray1(py).into_any(),
        }
    }

    fn get_sens<'py>(&self, py: Python<'py>) -> Vec<Bound<'py, PyUntypedArray2>> {
        self.sens.iter().map(|s| s.inner().to_pyarray2(py).into_any()).collect()
    }

    fn get_ts<'py>(&self, py: Python<'py>) -> Bound<'py, PyUntypedArray1> {
        self.ts.as_ref().unwrap().inner().to_pyarray1(py).into_any()
    }
    fn get_ys<'py>(&self, py: Python<'py>) -> Bound<'py, PyUntypedArray2> {
        self.ys.inner().to_pyarray2(py).into_any()
    }
}

