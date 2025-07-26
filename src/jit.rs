#[cfg(feature = "diffsol-cranelift")]
pub type JitModule = diffsol::CraneliftJitModule;

#[cfg(feature = "diffsol-llvm")]
pub type JitModule = diffsol::LlvmModule;