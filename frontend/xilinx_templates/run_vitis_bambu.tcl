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

#  Vitis (xsct) Automation Script for Bambu HLS on ZCU104
#
#  Same as run_vitis_app.tcl but adapted for Bambu:
#    - Copies xinference_bambu.h as xinference.h (provides HLS-compatible API)
#    - Copies model.h from output_c/<project>/ (not xilinx/<project>/src/)
#    - Patches xparameters.h if Bambu wrapper base address macro not found
#
#  Usage (from output_c/<bambu_project>/ directory):
#    xsct run_vitis_bambu.tcl [main_sd_path]
#
#  Output:
#    ./vitis_<project_name>/
#      +-- HW_<project_name>/        (platform)
#      +-- SW_<project_name>/        (application)
#      |   +-- src/
#      |       +-- main_sd.c
#      |       +-- xinference.h      (from xinference_bambu.h)
#      |       +-- model.h
#      +-- SW_<project_name>_system/ (system project)

# Setup
set proj_dir [file normalize [pwd]]
set proj_name [file tail $proj_dir]
# Builds live in backend_projects/<project>/bambu; name the workspace after <project>
if {$proj_name eq "bambu"} { set proj_name [file tail [file dirname $proj_dir]] }

# Auto-detect XSA file
set board_tag [expr {[info exists ::env(BOARD_TAG)] && $::env(BOARD_TAG) ne "" ? $::env(BOARD_TAG) : "zcu104"}]
set xsa_candidates [glob -nocomplain -directory $proj_dir vivado_*_${board_tag}/*.xsa]
if {[llength $xsa_candidates] == 0} {
    puts "ERROR: No .xsa file found in vivado_*_${board_tag}/"
    puts "       Run build_block_design_bambu.tcl first."
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

# Find main_sd.c
set main_sd_src ""
if {$argc >= 1} {
    set main_sd_src [file normalize [lindex $argv 0]]
} else {
    set search_paths [list \
        "$proj_dir/../../tools/main_sd.c" \
    ]
    foreach vdir [glob -nocomplain -directory $proj_dir vitis_*/SW_*/src/main_sd.c] {
        lappend search_paths $vdir
    }
    foreach p $search_paths {
        if {[file exists $p]} {
            set main_sd_src [file normalize $p]
            break
        }
    }
}

# Find xinference_bambu.h
set xinf_bambu_src ""
set xinf_search_paths [list \
    "$proj_dir/xinference_bambu.h" \
    "$proj_dir/../../tools/xinference_bambu.h" \
]
foreach p $xinf_search_paths {
    if {[file exists $p]} {
        set xinf_bambu_src [file normalize $p]
        break
    }
}

# Find model.h (in the Bambu project output dir)
set model_h_src ""
set model_h_search_paths [list \
    "$proj_dir/model.h" \
    "$proj_dir/src/model.h" \
]
foreach p $model_h_search_paths {
    if {[file exists $p]} {
        set model_h_src [file normalize $p]
        break
    }
}

puts "  Vitis Automation for Bambu HLS on ZCU104"
puts "  Project:           $proj_name"
puts "  XSA:               $xsa_path"
puts "  Workspace:         $ws_dir"
puts "  HW platform:       $hw_name"
puts "  SW app:            $sw_name"
if {$main_sd_src ne ""} {
    puts "  main_sd.c:         $main_sd_src"
} else {
    puts "  main_sd.c:         (not found - will need manual copy)"
}
if {$xinf_bambu_src ne ""} {
    puts "  xinference_bambu:  $xinf_bambu_src"
} else {
    puts "  xinference_bambu:  (not found - build may fail)"
}
if {$model_h_src ne ""} {
    puts "  model.h:           $model_h_src"
} else {
    puts "  model.h:           (not found)"
}

if {![file exists $xsa_path]} {
    puts "ERROR: XSA file not found: $xsa_path"
    exit 1
}

# Step 1: Set workspace
puts "\n===> Step 1/7: Setting workspace ..."
if {[file exists $ws_dir]} {
    puts "     Removing existing workspace: $ws_dir"
    file delete -force $ws_dir
}
setws $ws_dir

# Step 2: Create platform
puts "\n===> Step 2/7: Creating platform from XSA ..."

platform create -name $hw_name \
    -hw $xsa_path \
    -proc {psu_cortexa53_0} \
    -os {standalone} \
    -arch {64-bit} \
    -fsbl-target {psu_cortexa53_0} \
    -out $ws_dir

platform write

# Step 3: Configure xilffs
puts "\n===> Step 3/7: Configuring BSP (xilffs for SD card) ..."

platform active $hw_name

domain active {zynqmp_fsbl}
bsp reload

domain active {standalone_domain}
bsp reload

# Do not pin -ver: the xilffs version differs across Vitis releases and a stale
# pin fails the BSP; omitting it uses the version bundled with the running Vitis.
bsp setlib -name xilffs
bsp write
bsp reload
catch {bsp regenerate}

# Step 4: Build platform
puts "\n===> Step 4/7: Generating platform (FSBL + PMU + BSP) ..."
platform generate

# Step 5: Patch xparameters.h
puts "\n===> Step 5/7: Checking xparameters.h for Bambu wrapper base address ..."

# Find xparameters.h in the BSP
set xparam_candidates [glob -nocomplain "$ws_dir/$hw_name/*/standalone_domain/bsp/psu_cortexa53_0/include/xparameters.h"]
if {[llength $xparam_candidates] > 0} {
    set xparam_path [lindex $xparam_candidates 0]
    puts "     xparameters.h: $xparam_path"

    # Read xparameters.h content
    set fh [open $xparam_path r]
    set xparam_content [read $fh]
    close $fh

    # Check if any known Bambu wrapper macro exists
    set has_bambu_addr 0
    foreach pattern {BAMBU_WRAPPER_0 BAMBU_INFERENCE_AXI_WRAPPER_0} {
        if {[string match "*${pattern}*" $xparam_content]} {
            puts "     Found $pattern in xparameters.h"
            set has_bambu_addr 1
            break
        }
    }

    if {!$has_bambu_addr} {
        # Try to find the actual base address from any generic peripheral entry
        # that might correspond to the Bambu wrapper
        puts "     WARNING: No known Bambu wrapper macro found in xparameters.h"
        puts "     Searching for S_AXI_CONTROL entries ..."

        # Look for any S_AXI_CONTROL_BASEADDR entries
        set lines [split $xparam_content "\n"]
        foreach line $lines {
            if {[string match "*S_AXI_CONTROL_BASEADDR*" $line] ||
                [string match "*BAMBU*BASEADDR*" $line]} {
                puts "     Found: [string trim $line]"
            }
        }

        puts "     If build fails, grep xparameters.h for BASEADDR,"
        puts "     then define BAMBU_INFERENCE_BASEADDR in model.h or xinference.h"
    }
} else {
    puts "     WARNING: xparameters.h not found in BSP (platform may not be built yet)"
}

# Step 6: Create application
puts "\n===> Step 6/7: Creating application project ..."

app create -name $sw_name \
    -platform $hw_name \
    -domain {standalone_domain} \
    -proc {psu_cortexa53_0} \
    -os {standalone} \
    -lang {c} \
    -template {Empty Application}

# Copy source files
set sw_src_dir "$ws_dir/$sw_name/src"

# Copy main_sd.c
if {$main_sd_src ne "" && [file exists $main_sd_src]} {
    puts "     Copying main_sd.c from: $main_sd_src"
    file copy -force $main_sd_src "$sw_src_dir/main_sd.c"
} else {
    puts "WARNING: main_sd.c not found. Copy manually to: $sw_src_dir/"
}

# Copy xinference_bambu.h AS xinference.h (HLS-compatible API)
if {$xinf_bambu_src ne "" && [file exists $xinf_bambu_src]} {
    puts "     Copying xinference_bambu.h -> xinference.h"
    file copy -force $xinf_bambu_src "$sw_src_dir/xinference.h"
} else {
    puts "WARNING: xinference_bambu.h not found."
    puts "         Copy tools/xinference_bambu.h as xinference.h to: $sw_src_dir/"
}

# Copy model.h
if {$model_h_src ne "" && [file exists $model_h_src]} {
    puts "     Copying model.h from: $model_h_src"
    file copy -force $model_h_src "$sw_src_dir/model.h"
} else {
    puts "WARNING: model.h not found."
}

# Step 7: Build application
puts "\n===> Step 7/7: Building application ..."
app build -name $sw_name

# Done
set elf_path "$ws_dir/$sw_name/Debug/${sw_name}.elf"

puts "  Vitis Build Completed (Bambu)!"
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
