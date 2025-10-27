// Compile time JitModule type based on compilation settings. Currently Linux
// and macOS use LLVM and Windows uses Cranelift.

#[cfg(feature = "diffsol-cranelift")]
pub type JitModule = diffsol::CraneliftJitModule;

#[cfg(feature = "diffsol-llvm")]
pub type JitModule = diffsol::LlvmModule;
