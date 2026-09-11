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

# Vivado Block Design for Bambu HLS on ZCU104 (BRAM-buffered)
#
# Architecture:
#   PS M_AXI_HPM0_FPD -> AXI Interconnect -> {Bambu wrapper, GPIO}
#   No AXI master needed - DUT uses local BRAM for input/output.
#
# Usage:
#   vivado -mode batch -source build_block_design_bambu.tcl [-tclargs <project_name>]

# Basic paths and names
set root_dir   [file normalize [pwd]]

if {$argc > 0} {
    set user_proj_name [lindex $argv 0]
} elseif {[info exists ::env(PROJECT_NAME)]} {
    set user_proj_name $::env(PROJECT_NAME)
} else {
    set user_proj_name "bambu_inference"
}

set board_tag  [expr {[info exists ::env(BOARD_TAG)] && $::env(BOARD_TAG) ne "" ? $::env(BOARD_TAG) : "zcu104"}]
set proj_name  "${user_proj_name}_${board_tag}"
set proj_dir   [file normalize "$root_dir/vivado_${proj_name}"]

# Device and board
set part_name   [expr {[info exists ::env(FPGA_PART)]  && $::env(FPGA_PART)  ne "" ? $::env(FPGA_PART)  : "xczu7ev-ffvc1156-2-e"}]
# board match string for get_board_parts; override BOARD_MATCH to target another board.
set board_match [expr {[info exists ::env(BOARD_MATCH)] && $::env(BOARD_MATCH) ne "" ? $::env(BOARD_MATCH) : "*zcu104*"}]

# Newer Vivado does not bundle the ZCU104 board files; honor BOARD_REPO_PATHS so
# users can point Vivado at their installed board files. board_part is resolved
# dynamically after project creation so the script is version-agnostic.
if {[info exists ::env(BOARD_REPO_PATHS)] && $::env(BOARD_REPO_PATHS) ne ""} {
    set_param board.repoPaths $::env(BOARD_REPO_PATHS)
}

# Block Design cell names
set bd_name      design_1
set ps_name      zynq_ultra_ps_e_0
set wrapper_name bambu_wrapper_0
set gpio_name    axi_gpio_0

puts "===> user_proj_name = $user_proj_name"
puts "===> root_dir       = $root_dir"
puts "===> proj_dir       = $proj_dir"

# Step 1: Create Vivado project
create_project -force $proj_name $proj_dir -part $part_name

# Attach whatever ZCU104 board revision is available (version-agnostic).
set board_part [lindex [get_board_parts -quiet $board_match] 0]
if {$board_part ne ""} {
    set_property board_part $board_part [current_project]
    puts "===> Using board_part $board_part"
} else {
    puts "WARNING: no ZCU104 board part found; only device part set. Set BOARD_REPO_PATHS to your zcu104 board files."
}

# Add Bambu RTL sources (inference.v + AXI wrapper)
add_files -norecurse [list \
    [file normalize "$root_dir/inference.v"] \
    [file normalize "$root_dir/bambu_inference_axi_wrapper.v"] \
]
update_compile_order -fileset sources_1

# Step 2: Create Block Design
create_bd_design $bd_name

# 2.1 Add Zynq UltraScale+ MPSoC (PS; IP version resolved from catalog)
set ps_vlnv [lindex [lsort -dictionary [get_ipdefs -all -filter {NAME == zynq_ultra_ps_e}]] end]
create_bd_cell -type ip -vlnv $ps_vlnv $ps_name

# Apply the ZCU104 board preset
apply_bd_automation -rule xilinx.com:bd_rule:zynq_ultra_ps_e \
    -config {apply_board_preset "1"} \
    [get_bd_cells $ps_name]

# Configure DDR preset and disable unused PS interfaces
set_property -dict [list \
    CONFIG.SUBPRESET1                {DDR4_MICRON_MT40A256M16GE_083E} \
    CONFIG.PSU__USE__M_AXI_GP1       {0} \
] [get_bd_cells $ps_name]

# 2.2 Add Bambu wrapper as module reference
create_bd_cell -type module -reference bambu_inference_axi_wrapper $wrapper_name

# 2.3 Add AXI GPIO for the 4-bit LED output
set gpio_vlnv [lindex [lsort -dictionary [get_ipdefs -all -filter {NAME == axi_gpio}]] end]
create_bd_cell -type ip -vlnv $gpio_vlnv $gpio_name

set_property -dict [list \
    CONFIG.C_GPIO_WIDTH         {4} \
    CONFIG.GPIO_BOARD_INTERFACE {led_4bits} \
    CONFIG.C_ALL_OUTPUTS        {1} \
] [get_bd_cells $gpio_name]

# Step 3: AXI connection automation

# 3.1 Connect GPIO to LED board interface
apply_bd_automation -rule xilinx.com:bd_rule:board \
    -config { Board_Interface {led_4bits ( LED ) } Manual_Source {Auto} } \
    [get_bd_intf_pins $gpio_name/GPIO]

# 3.2 Connect AXI for GPIO: M_AXI_HPM0_FPD -> axi_gpio_0/S_AXI
apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config { \
    Clk_master {Auto} \
    Clk_slave  {Auto} \
    Clk_xbar   {Auto} \
    Master     {/zynq_ultra_ps_e_0/M_AXI_HPM0_FPD} \
    Slave      {/axi_gpio_0/S_AXI} \
    ddr_seg    {Auto} \
    intc_ip    {New AXI Interconnect} \
    master_apm {0} \
} [get_bd_intf_pins axi_gpio_0/S_AXI]

# 3.3 Connect AXI for Bambu wrapper: M_AXI_HPM0_FPD -> wrapper/S_AXI_CONTROL
# Wrapper has 18-bit address (256KB), Vivado auto-assigns address range
apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config { \
    Clk_master {Auto} \
    Clk_slave  {Auto} \
    Clk_xbar   {Auto} \
    Master     {/zynq_ultra_ps_e_0/M_AXI_HPM0_FPD} \
    Slave      {/bambu_wrapper_0/S_AXI_CONTROL} \
    ddr_seg    {Auto} \
    intc_ip    {/ps8_0_axi_periph} \
    master_apm {0} \
} [get_bd_intf_pins bambu_wrapper_0/S_AXI_CONTROL]

# Regenerate layout for clean diagram
regenerate_bd_layout

# Step 4: Validate and save Block Design
validate_bd_design
save_bd_design

# Step 5: Generate HDL wrapper
set bd_file [get_files "$proj_dir/$proj_name.srcs/sources_1/bd/$bd_name/$bd_name.bd"]

make_wrapper -files $bd_file -top

set wrapper_file "$proj_dir/$proj_name.srcs/sources_1/bd/$bd_name/hdl/${bd_name}_wrapper.v"
add_files -norecurse $wrapper_file
set_property top ${bd_name}_wrapper [current_fileset]

update_compile_order -fileset sources_1

# Step 6: Synthesis, Implementation, Bitstream
# Parallelize OOC IP synthesis across cores. This tcl runs on the build
# machine, so nproc reads its actual core count at run time; cap at 8 since
# the block design has only ~10 OOC runs and more jobs stop helping.
if {[catch {exec nproc} n_cpu] || ![string is integer -strict $n_cpu]} {
    set n_cpu 8
}
set n_jobs [expr {$n_cpu < 8 ? $n_cpu : 8}]

launch_runs synth_1 -jobs $n_jobs
wait_on_run synth_1

launch_runs impl_1 -to_step write_bitstream -jobs $n_jobs
wait_on_run impl_1

# Step 7: Export hardware platform (XSA with bitstream)
set xsa_file "$proj_dir/${user_proj_name}_${board_tag}.xsa"
write_hw_platform -fixed -force -include_bit -file $xsa_file

puts "===> Vivado build completed."
puts "     Project directory: $proj_dir"
puts "     Bitstream:         $proj_dir/$proj_name.runs/impl_1/${bd_name}_wrapper.bit"
puts "     XSA:               $xsa_file"

exit
