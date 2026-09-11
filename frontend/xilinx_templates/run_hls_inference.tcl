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

#  Vitis HLS Automation Script
#  - Creates an HLS project
#  - Imports all C/C++/header files in ./src
#  - Sets the top function
#  - Sets FPGA part (ZCU104: xczu7ev-ffvc1156-2-e)
#  - Runs C synthesis
#  - Exports the synthesized module as an IP

# Project name
set proj_name  inference_hls

# Root directory (directory containing this script)
set root_dir   [file normalize [file dirname [info script]]]
cd $root_dir

# Source directory containing .c/.h files
set src_dir    [file normalize "$root_dir/src"]

# Top-level function name for HLS
set top_name   inference

# Target FPGA part (ZCU104 default; override via FPGA_PART to match the Vivado part).
set part_name  [expr {[info exists ::env(FPGA_PART)] && $::env(FPGA_PART) ne "" ? $::env(FPGA_PART) : "xczu7ev-ffvc1156-2-e"}]

# Output directory for exported IP
set ip_out_dir [file normalize "$root_dir/hls_ip"]

puts "===> HLS root_dir   = $root_dir"
puts "===> HLS src_dir    = $src_dir"
puts "===> HLS ip_out_dir = $ip_out_dir"
puts "===> HLS top_name   = $top_name"
puts "===> HLS part_name  = $part_name"

# Create project
open_project -reset $proj_name

# Set top-level function
set_top $top_name

# Add C/C++ source files and headers
set src_files [glob -nocomplain -directory $src_dir *.{c,cpp,cc,C,h,hpp}]
if {[llength $src_files] == 0} {
    puts "ERROR: No source files found in $src_dir"
    exit 1
}

puts "===> Adding source files:"
puts $src_files

add_files $src_files

# Create solution and set FPGA part/clock
open_solution -reset solution1

# Select FPGA part
set_part $part_name

# Default 100 MHz clock
create_clock -period 10 -name default

# Optional per-project HLS op directives (converter writes hls_directives.tcl only for
# --mul-impl-fabric; absent otherwise). Must precede csynth_design to take effect.
if {[file exists "$root_dir/hls_directives.tcl"]} {
    puts "===> Sourcing extra HLS directives: $root_dir/hls_directives.tcl"
    source "$root_dir/hls_directives.tcl"
}

# C synthesis
csynth_design

# Export design as IP (Verilog RTL)
config_export -format ip_catalog -rtl verilog

file mkdir $ip_out_dir

# Remove stale IP directory before export to prevent Vivado from
# using an old cached version instead of the newly synthesized one
set stale_ip_dir "$ip_out_dir/inference_ip"
if {[file exists $stale_ip_dir]} {
    puts "===> Removing stale IP directory: $stale_ip_dir"
    file delete -force $stale_ip_dir
}

export_design \
    -format ip_catalog \
    -rtl verilog \
    -output $ip_out_dir

# Extract the exported zip to create the IP catalog directory
# (export_design only creates the zip, Vivado needs the extracted directory)
set export_zip "$ip_out_dir/export.zip"
if {[file exists $export_zip]} {
    set ip_extract_dir "$ip_out_dir/inference_ip"
    if {![file exists $ip_extract_dir]} {
        file mkdir $ip_extract_dir
        puts "===> Extracting $export_zip to $ip_extract_dir"
        exec unzip -o $export_zip -d $ip_extract_dir
    }
}

puts "===> HLS export completed. IP exported to:"
puts "$ip_out_dir"

exit
