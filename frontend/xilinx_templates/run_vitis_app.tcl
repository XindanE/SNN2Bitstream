# This file is part of SNN2Bitstream.
# Copyright (C) 2026 Xindan Zhang, Sorbonne Université, CNRS, LIP6

# SNN2Bitstream is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# SNN2Bitstream is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

#  Vitis (xsct) Automation Script for ZCU104
#
#  Creates a Vitis workspace with:
#    - Hardware platform from XSA (with xilffs BSP)
#    - Application project with main_sd.c
#    - Builds the ELF binary
#
#  Usage (from project directory, e.g. xilinx/nmnist_fcn_t10_h128_qat_ft/):
#    xsct run_vitis_app.tcl [main_sd_path]
#
#  The script auto-detects:
#    - XSA: vivado_*_zcu104/*.xsa
#    - Project name: from current directory name (its parent when run from backend_projects/<project>/xilinx)
#    - main_sd.c: from argument, or previous vitis project, or tools/
#
#  Output:
#    ./vitis_<project_name>/
#      +-- HW_<project_name>/        (platform)
#      +-- SW_<project_name>/        (application)
#      |   +-- src/main_sd.c
#      +-- SW_<project_name>_system/ (system project)

# Setup
set proj_dir [file normalize [pwd]]
set proj_name [file tail $proj_dir]
# Builds live in backend_projects/<project>/xilinx; name the workspace after <project>
if {$proj_name eq "xilinx"} { set proj_name [file tail [file dirname $proj_dir]] }

# Auto-detect XSA file (prefer one matching project name). BOARD_TAG must match the
# one the Vivado build used, otherwise the project directory name will not match.
set board_tag [expr {[info exists ::env(BOARD_TAG)] && $::env(BOARD_TAG) ne "" ? $::env(BOARD_TAG) : "zcu104"}]
set xsa_candidates [glob -nocomplain -directory $proj_dir vivado_*_${board_tag}/*.xsa]
if {[llength $xsa_candidates] == 0} {
    puts "ERROR: No .xsa file found in vivado_*_${board_tag}/"
    puts "       Run Vivado build first."
    exit 1
}
set xsa_path ""
foreach candidate $xsa_candidates {
    if {[string match "*${proj_name}*" $candidate]} {
        set xsa_path $candidate
        break
    }
}
if {$xsa_path eq ""} {
    set xsa_path [lindex $xsa_candidates end]
}

# Names
set ws_dir   [file normalize "$proj_dir/vitis_${proj_name}"]
set hw_name  "HW_${proj_name}"
set sw_name  "SW_${proj_name}"

# Find main_sd.c: cmdline arg > local SW project > tools/
set main_sd_src ""
if {$argc >= 1} {
    set main_sd_src [file normalize [lindex $argv 0]]
} else {
    # Look for main_sd.c in known locations (priority order)
    set search_paths [list \
        "$proj_dir/SW_${proj_name}/src/main_sd.c" \
        "$proj_dir/../../tools/main_sd.c" \
    ]
    # Also search in any existing vitis_* workspace
    foreach vdir [glob -nocomplain -directory $proj_dir vitis_*/SW_*/src/main_sd.c] {
        lappend search_paths $vdir
    }
    # Search in sibling vitis_* directories
    set parent [file dirname $proj_dir]
    foreach vdir [glob -nocomplain -directory $parent vitis_*/SW_*/src/main_sd.c] {
        lappend search_paths $vdir
    }
    foreach p $search_paths {
        if {[file exists $p]} {
            set main_sd_src [file normalize $p]
            break
        }
    }
}

puts "  Vitis Automation for ZCU104"
puts "  Project:      $proj_name"
puts "  XSA:          $xsa_path"
puts "  Workspace:    $ws_dir"
puts "  HW platform:  $hw_name"
puts "  SW app:       $sw_name"
if {$main_sd_src ne ""} {
    puts "  main_sd.c:    $main_sd_src"
} else {
    puts "  main_sd.c:    (not found - will need manual copy)"
}

if {![file exists $xsa_path]} {
    puts "ERROR: XSA file not found: $xsa_path"
    exit 1
}

# Step 1: Set workspace
puts "\n===> Step 1/6: Setting workspace ..."
if {[file exists $ws_dir]} {
    puts "     Removing existing workspace: $ws_dir"
    file delete -force $ws_dir
}
setws $ws_dir

# Step 2: Create platform
puts "\n===> Step 2/6: Creating platform from XSA ..."

platform create -name $hw_name \
    -hw $xsa_path \
    -proc {psu_cortexa53_0} \
    -os {standalone} \
    -arch {64-bit} \
    -fsbl-target {psu_cortexa53_0} \
    -out $ws_dir

platform write

# Step 3: Configure xilffs
puts "\n===> Step 3/6: Configuring BSP (xilffs for SD card) ..."

platform active $hw_name

domain active {zynqmp_fsbl}
bsp reload

domain active {standalone_domain}
bsp reload

# Add xilffs library (FAT filesystem for SD card access).
# Do not pin -ver: the xilffs version differs across Vitis releases and a stale
# pin fails the BSP; omitting it uses the version bundled with the running Vitis.
bsp setlib -name xilffs
bsp write
bsp reload
catch {bsp regenerate}

# Step 4: Build platform
puts "\n===> Step 4/6: Generating platform (FSBL + PMU + BSP) ..."
platform generate

# Step 5: Create application
puts "\n===> Step 5/6: Creating application project ..."

app create -name $sw_name \
    -platform $hw_name \
    -domain {standalone_domain} \
    -proc {psu_cortexa53_0} \
    -os {standalone} \
    -lang {c} \
    -template {Empty Application}

# Copy main_sd.c
set sw_src_dir "$ws_dir/$sw_name/src"
if {$main_sd_src ne "" && [file exists $main_sd_src]} {
    puts "     Copying main_sd.c from: $main_sd_src"
    file copy -force $main_sd_src "$sw_src_dir/main_sd.c"
} else {
    puts "WARNING: main_sd.c not found."
    puts "         Copy it manually to: $sw_src_dir/"
    puts "         Then rebuild with: app build -name $sw_name"
}

# Copy model.h (needed by main_sd.c for Phase 3 auto-adaptive type macros)
set model_h_src "$proj_dir/src/model.h"
if {[file exists $model_h_src]} {
    puts "     Copying model.h from: $model_h_src"
    file copy -force $model_h_src "$sw_src_dir/model.h"
} else {
    puts "WARNING: model.h not found at $model_h_src"
}

# Vitis HLS >= 2022 appends "_r" to m_axi pointer-arg register names; tell
# main_sd.c via HLS_2022_PLUS so it picks the matching driver symbols.
if {[regexp {(20\d\d)\.\d} [version] -> vitis_year] && $vitis_year >= 2022} {
    puts "===> Vitis $vitis_year (>= 2022): defining HLS_2022_PLUS for the app"
    app config -name $sw_name define-compiler-symbols HLS_2022_PLUS
}

# Step 6: Build application
puts "\n===> Step 6/6: Building application ..."
app build -name $sw_name

# Done
set elf_path "$ws_dir/$sw_name/Debug/${sw_name}.elf"

puts "  Vitis Build Completed!"
puts "  Workspace:  $ws_dir"
puts "  Platform:   $ws_dir/$hw_name"
puts "  App:        $ws_dir/$sw_name"
if {[file exists $elf_path]} {
    puts "  ELF:        $elf_path"
} else {
    puts "  ELF:        (build may have failed - check above)"
}
puts ""
puts "  Next steps:"
puts "    # Open Vitis GUI for serial window + run:"
puts "    vitis -workspace $ws_dir"
puts ""
puts "    # Or use terminal serial monitor:"
puts "    picocom -b 115200 /dev/ttyUSB1"

exit
