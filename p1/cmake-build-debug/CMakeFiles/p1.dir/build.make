# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.12

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = "/cygdrive/c/Users/Miss Guo/.CLion2018.2/system/cygwin_cmake/bin/cmake.exe"

# The command to remove a file.
RM = "/cygdrive/c/Users/Miss Guo/.CLion2018.2/system/cygwin_cmake/bin/cmake.exe" -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = "/cygdrive/c/Users/Miss Guo/Desktop/ECE6122-Adv Prog Techniques/Homework/p1"

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = "/cygdrive/c/Users/Miss Guo/Desktop/ECE6122-Adv Prog Techniques/Homework/p1/cmake-build-debug"

# Include any dependencies generated for this target.
include CMakeFiles/p1.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/p1.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/p1.dir/flags.make

CMakeFiles/p1.dir/src/array.cc.o: CMakeFiles/p1.dir/flags.make
CMakeFiles/p1.dir/src/array.cc.o: ../src/array.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir="/cygdrive/c/Users/Miss Guo/Desktop/ECE6122-Adv Prog Techniques/Homework/p1/cmake-build-debug/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/p1.dir/src/array.cc.o"
	/usr/bin/c++.exe  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/p1.dir/src/array.cc.o -c "/cygdrive/c/Users/Miss Guo/Desktop/ECE6122-Adv Prog Techniques/Homework/p1/src/array.cc"

CMakeFiles/p1.dir/src/array.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/p1.dir/src/array.cc.i"
	/usr/bin/c++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E "/cygdrive/c/Users/Miss Guo/Desktop/ECE6122-Adv Prog Techniques/Homework/p1/src/array.cc" > CMakeFiles/p1.dir/src/array.cc.i

CMakeFiles/p1.dir/src/array.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/p1.dir/src/array.cc.s"
	/usr/bin/c++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S "/cygdrive/c/Users/Miss Guo/Desktop/ECE6122-Adv Prog Techniques/Homework/p1/src/array.cc" -o CMakeFiles/p1.dir/src/array.cc.s

CMakeFiles/p1.dir/src/simple_string.cc.o: CMakeFiles/p1.dir/flags.make
CMakeFiles/p1.dir/src/simple_string.cc.o: ../src/simple_string.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir="/cygdrive/c/Users/Miss Guo/Desktop/ECE6122-Adv Prog Techniques/Homework/p1/cmake-build-debug/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/p1.dir/src/simple_string.cc.o"
	/usr/bin/c++.exe  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/p1.dir/src/simple_string.cc.o -c "/cygdrive/c/Users/Miss Guo/Desktop/ECE6122-Adv Prog Techniques/Homework/p1/src/simple_string.cc"

CMakeFiles/p1.dir/src/simple_string.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/p1.dir/src/simple_string.cc.i"
	/usr/bin/c++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E "/cygdrive/c/Users/Miss Guo/Desktop/ECE6122-Adv Prog Techniques/Homework/p1/src/simple_string.cc" > CMakeFiles/p1.dir/src/simple_string.cc.i

CMakeFiles/p1.dir/src/simple_string.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/p1.dir/src/simple_string.cc.s"
	/usr/bin/c++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S "/cygdrive/c/Users/Miss Guo/Desktop/ECE6122-Adv Prog Techniques/Homework/p1/src/simple_string.cc" -o CMakeFiles/p1.dir/src/simple_string.cc.s

CMakeFiles/p1.dir/main.cc.o: CMakeFiles/p1.dir/flags.make
CMakeFiles/p1.dir/main.cc.o: ../main.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir="/cygdrive/c/Users/Miss Guo/Desktop/ECE6122-Adv Prog Techniques/Homework/p1/cmake-build-debug/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/p1.dir/main.cc.o"
	/usr/bin/c++.exe  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/p1.dir/main.cc.o -c "/cygdrive/c/Users/Miss Guo/Desktop/ECE6122-Adv Prog Techniques/Homework/p1/main.cc"

CMakeFiles/p1.dir/main.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/p1.dir/main.cc.i"
	/usr/bin/c++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E "/cygdrive/c/Users/Miss Guo/Desktop/ECE6122-Adv Prog Techniques/Homework/p1/main.cc" > CMakeFiles/p1.dir/main.cc.i

CMakeFiles/p1.dir/main.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/p1.dir/main.cc.s"
	/usr/bin/c++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S "/cygdrive/c/Users/Miss Guo/Desktop/ECE6122-Adv Prog Techniques/Homework/p1/main.cc" -o CMakeFiles/p1.dir/main.cc.s

# Object files for target p1
p1_OBJECTS = \
"CMakeFiles/p1.dir/src/array.cc.o" \
"CMakeFiles/p1.dir/src/simple_string.cc.o" \
"CMakeFiles/p1.dir/main.cc.o"

# External object files for target p1
p1_EXTERNAL_OBJECTS =

p1.exe: CMakeFiles/p1.dir/src/array.cc.o
p1.exe: CMakeFiles/p1.dir/src/simple_string.cc.o
p1.exe: CMakeFiles/p1.dir/main.cc.o
p1.exe: CMakeFiles/p1.dir/build.make
p1.exe: CMakeFiles/p1.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir="/cygdrive/c/Users/Miss Guo/Desktop/ECE6122-Adv Prog Techniques/Homework/p1/cmake-build-debug/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_4) "Linking CXX executable p1.exe"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/p1.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/p1.dir/build: p1.exe

.PHONY : CMakeFiles/p1.dir/build

CMakeFiles/p1.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/p1.dir/cmake_clean.cmake
.PHONY : CMakeFiles/p1.dir/clean

CMakeFiles/p1.dir/depend:
	cd "/cygdrive/c/Users/Miss Guo/Desktop/ECE6122-Adv Prog Techniques/Homework/p1/cmake-build-debug" && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" "/cygdrive/c/Users/Miss Guo/Desktop/ECE6122-Adv Prog Techniques/Homework/p1" "/cygdrive/c/Users/Miss Guo/Desktop/ECE6122-Adv Prog Techniques/Homework/p1" "/cygdrive/c/Users/Miss Guo/Desktop/ECE6122-Adv Prog Techniques/Homework/p1/cmake-build-debug" "/cygdrive/c/Users/Miss Guo/Desktop/ECE6122-Adv Prog Techniques/Homework/p1/cmake-build-debug" "/cygdrive/c/Users/Miss Guo/Desktop/ECE6122-Adv Prog Techniques/Homework/p1/cmake-build-debug/CMakeFiles/p1.dir/DependInfo.cmake" --color=$(COLOR)
.PHONY : CMakeFiles/p1.dir/depend

