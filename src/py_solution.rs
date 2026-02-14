use diffsol::{
    BdfState, DefaultDenseMatrix, DenseMatrix, MatrixCommon, OdeSolverState, RkState, Vector,
    VectorCommon, VectorHost, VectorViewMut,
};
use num_traits::FromPrimitive;
use pyo3::{Bound, Python};
use std::any::Any;

use crate::error::PyDiffsolError;
use crate::py_convert::{MatrixToPy, VectorToPy};
use crate::py_types::{PyUntypedArray1, PyUntypedArray2};

pub(crate) trait PySolution: Any {
    fn get_ys<'py>(&self, py: Python<'py>) -> Bound<'py, PyUntypedArray2>;
    fn get_ts<'py>(&self, py: Python<'py>) -> Bound<'py, PyUntypedArray1>;
    fn get_sens<'py>(&self, py: Python<'py>) -> Vec<Bound<'py, PyUntypedArray2>>;
    fn set_state_y(&mut self, y: &[f64]);
    fn get_state_y<'py>(&self, py: Python<'py>) -> Bound<'py, PyUntypedArray1>;
}

impl dyn PySolution + '_ {
    pub(crate) fn downcast_typed_solution<V>(&self) -> Result<&GenericPySolution<V>, PyDiffsolError>
    where
        V: Vector + DefaultDenseMatrix + 'static,
    {
        (self as &dyn Any)
            .downcast_ref::<GenericPySolution<V>>()
            .ok_or_else(|| {
                PyDiffsolError::Conversion(
                    "Provided Solution type is incompatible with this Ode instance".to_string(),
                )
            })
    }

    pub(crate) fn downcast_typed_solution_mut<V>(
        &mut self,
    ) -> Result<&mut GenericPySolution<V>, PyDiffsolError>
    where
        V: Vector + DefaultDenseMatrix + 'static,
    {
        (self as &mut dyn Any)
            .downcast_mut::<GenericPySolution<V>>()
            .ok_or_else(|| {
                PyDiffsolError::Conversion(
                    "Provided Solution type is incompatible with this Ode instance".to_string(),
                )
            })
    }
}

#[derive(Clone)]
pub(crate) enum GenericPyState<V: Vector + DefaultDenseMatrix> {
    Bdf(BdfState<V>),
    Rk(RkState<V>),
}

pub(crate) struct GenericPySolution<V: Vector + DefaultDenseMatrix> {
    state: Option<GenericPyState<V>>,
    ys: <V as DefaultDenseMatrix>::M,
    ts: Vec<V::T>,
    sens: Vec<<V as DefaultDenseMatrix>::M>,
}

impl<V: Vector + DefaultDenseMatrix> GenericPySolution<V> {
    pub(crate) fn new(
        state: GenericPyState<V>,
        ys: <V as DefaultDenseMatrix>::M,
        ts: Vec<V::T>,
        sens: Vec<<V as DefaultDenseMatrix>::M>,
    ) -> Self {
        Self {
            state: Some(state),
            ys,
            ts,
            sens,
        }
    }

    fn current_state(&self) -> &GenericPyState<V> {
        self.state
            .as_ref()
            .expect("solution current state missing unexpectedly")
    }

    fn current_state_mut(&mut self) -> &mut GenericPyState<V> {
        self.state
            .as_mut()
            .expect("solution current state missing unexpectedly")
    }

    pub(crate) fn state_clone(&self) -> Result<GenericPyState<V>, PyDiffsolError> {
        self.state
            .as_ref()
            .cloned()
            .ok_or_else(|| PyDiffsolError::Conversion("Solution current state missing".to_string()))
    }

    pub(crate) fn append(
        &mut self,
        state: GenericPyState<V>,
        ys: <V as DefaultDenseMatrix>::M,
        ts: Vec<V::T>,
        sens: Vec<<V as DefaultDenseMatrix>::M>,
    ) -> Result<(), String> {
        if self.ys.nrows() != ys.nrows() {
            return Err(format!(
                "Cannot append ys with mismatched rows ({} vs {})",
                self.ys.nrows(),
                ys.nrows()
            ));
        }

        // Validate sensitivity dimensions before mutating any buffers.
        if self.sens.is_empty() {
            // no-op validation
        } else if !sens.is_empty() {
            if self.sens.len() != sens.len() {
                return Err(format!(
                    "Cannot append sens with mismatched length ({} vs {})",
                    self.sens.len(),
                    sens.len()
                ));
            }
            for (dst, src) in self.sens.iter_mut().zip(sens.iter()) {
                if dst.nrows() != src.nrows() {
                    return Err(format!(
                        "Cannot append sens with mismatched rows ({} vs {})",
                        dst.nrows(),
                        src.nrows()
                    ));
                }
            }
        }

        append_matrix_columns(&mut self.ys, &ys);
        if self.sens.is_empty() {
            self.sens = sens;
        } else if !sens.is_empty() {
            for (dst, src) in self.sens.iter_mut().zip(sens.iter()) {
                append_matrix_columns(dst, src);
            }
        }
        self.ts.extend(ts);
        self.state = Some(state);
        Ok(())
    }
}

fn append_matrix_columns<M: DenseMatrix>(dst: &mut M, src: &M) {
    let old_cols = dst.ncols();
    let add_cols = src.ncols();
    dst.resize_cols(old_cols + add_cols);
    for j in 0..add_cols {
        dst.column_mut(old_cols + j).copy_from_view(&src.column(j));
    }
}

fn copy_slice_to_vec<V: VectorHost>(state: &mut V, y: &[f64]) {
    for (yi, &y_val) in state.as_mut_slice().iter_mut().zip(y.iter()) {
        *yi = V::T::from_f64(y_val).unwrap();
    }
}

impl<V: VectorHost + DefaultDenseMatrix + 'static> PySolution for GenericPySolution<V>
where
    for<'b> <V as VectorCommon>::Inner: VectorToPy<'b, V::T>,
    for<'b> <<V as DefaultDenseMatrix>::M as MatrixCommon>::Inner: MatrixToPy<'b, V::T>,
{
    fn set_state_y(&mut self, y: &[f64]) {
        match self.current_state_mut() {
            GenericPyState::Bdf(state) => copy_slice_to_vec(state.as_mut().y, y),
            GenericPyState::Rk(state) => copy_slice_to_vec(state.as_mut().y, y),
        }
    }

    fn get_state_y<'py>(&self, py: Python<'py>) -> Bound<'py, PyUntypedArray1> {
        match self.current_state() {
            GenericPyState::Bdf(state) => state.as_ref().y.inner().to_pyarray1(py).into_any(),
            GenericPyState::Rk(state) => state.as_ref().y.inner().to_pyarray1(py).into_any(),
        }
    }

    fn get_sens<'py>(&self, py: Python<'py>) -> Vec<Bound<'py, PyUntypedArray2>> {
        self.sens
            .iter()
            .map(|s| s.inner().to_pyarray2(py).into_any())
            .collect()
    }

    fn get_ts<'py>(&self, py: Python<'py>) -> Bound<'py, PyUntypedArray1> {
        let ctx = match self.current_state() {
            GenericPyState::Bdf(state) => state.as_ref().y.context().clone(),
            GenericPyState::Rk(state) => state.as_ref().y.context().clone(),
        };
        V::from_slice(&self.ts, ctx)
            .inner()
            .to_pyarray1(py)
            .into_any()
    }

    fn get_ys<'py>(&self, py: Python<'py>) -> Bound<'py, PyUntypedArray2> {
        self.ys.inner().to_pyarray2(py).into_any()
    }
}
