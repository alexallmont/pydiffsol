#!/usr/bin/env bash

# The maturin build has some unexpected dynamic dependencies that are likely not
# installed for macos users. This script copies the homebrew libzstd and LLVM
# libunwind - that were used to build the wheel - back into the generated wheel
# archive. It fixes rpaths for the .so file so dylibs are found relative to it.

# Unzip the build wheel, assumes dist folder in CI script
cd dist
WHEEL_FILE=$(find . -name "pydiffsol*.whl" -exec basename {} \;)
unzip $WHEEL_FILE -d wheel_fix_rpath

# Copy dependencies into unzip dir
cp $LLVM_PATH/lib/libunwind.1.dylib wheel_fix_rpath/pydiffsol
cp /opt/homebrew/opt/zstd/lib/libzstd.1.dylib wheel_fix_rpath/pydiffsol

# Fix rpath of wheels .so file so it references local dylibs
SO_FILE=$(find wheel_fix_rpath -name "*.so")
install_name_tool -change @rpath/libc++.1.dylib /usr/lib/libc++.1.dylib $SO_FILE
install_name_tool -change @rpath/libunwind.1.dylib @loader_path/libunwind.1.dylib $SO_FILE
install_name_tool -change /opt/homebrew/opt/zstd/lib/libzstd.1.dylib @loader_path/libzstd.1.dylib $SO_FILE

# Re-zip wheel and remove working dir (otherwise it is copied with artifacts)
cd wheel_fix_rpath
zip -r ../$WHEEL_FILE .
cd ..
rm -rf wheel_fix_rpath