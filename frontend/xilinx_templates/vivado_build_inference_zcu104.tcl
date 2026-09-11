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

# Vivado Automation Script for ZCU104 + HLS "inference" IP
#
# Steps:
#   1. Unzip HLS export.zip into an IP repository directory
#   2. Create a Vivado project for ZCU104 (xczu7ev-ffvc1156-2-e)
#   3. Create a Block Design:
#       - Add Zynq UltraScale+ MPSoC
#       - Apply board preset for ZCU104
#       - Set DDR preset to DDR4_MICRON_MT40A256M16GE_083E
#       - Enable S_AXI_GP0
#       - Add HLS IP "inference"
#       - Add AXI GPIO and bind it to the 4-bit LED board interface
#       - Run board and AXI connection automation
#   4. Generate HDL wrapper
#   5. Run synthesis, implementation, bitstream
#   6. Export hardware XSA (including bitstream)

# Basic paths and names
set root_dir   [file normalize [pwd]]

# Get project name from command line arg (-tclargs) or environment variable or default
# Usage: vivado -mode batch -source this.tcl -tclargs <project_name>
if {$argc > 0} {
    set user_proj_name [lindex $argv 0]
} elseif {[info exists ::env(PROJECT_NAME)]} {
    set user_proj_name $::env(PROJECT_NAME)
} else {
    set user_proj_name "inference"
}

# Board suffix for the generated project name; run_xilinx.sh exports BOARD_TAG so the
# shell and Tcl agree on the paths when targeting a board other than the ZCU104.
set board_tag  [expr {[info exists ::env(BOARD_TAG)] && $::env(BOARD_TAG) ne "" ? $::env(BOARD_TAG) : "zcu104"}]
set proj_name  "${user_proj_name}_${board_tag}"
set proj_dir   [file normalize "$root_dir/vivado_${proj_name}"]

puts "===> user_proj_name = $user_proj_name"

# HLS export from Vitis HLS
set ip_zip     [file normalize "$root_dir/hls_ip/export.zip"]
set ip_repo    [file normalize "$root_dir/hls_ip/inference_ip"]

# Block Design and IP instance names
set bd_name    design_1
set ps_name    zynq_ultra_ps_e_0
set hls_ip_vlnv "xilinx.com:hls:inference:1.0"
set hls_ip_name inference_0
set gpio_name  axi_gpio_0

# Device. Only the device part is pinned (stable across Vivado versions); the
# board_part revision is resolved dynamically below so the script is portable.
set part_name   [expr {[info exists ::env(FPGA_PART)]  && $::env(FPGA_PART)  ne "" ? $::env(FPGA_PART)  : "xczu7ev-ffvc1156-2-e"}]
# board match string for get_board_parts; override BOARD_MATCH to target another board.
set board_match [expr {[info exists ::env(BOARD_MATCH)] && $::env(BOARD_MATCH) ne "" ? $::env(BOARD_MATCH) : "*zcu104*"}]

# Newer Vivado does not bundle the ZCU104 board files. Honor BOARD_REPO_PATHS so
# users can point Vivado at their installed board files (e.g. the Xilinx Board
# Store); on 2020.2 the bundled board files are found without it.
if {[info exists ::env(BOARD_REPO_PATHS)] && $::env(BOARD_REPO_PATHS) ne ""} {
    set_param board.repoPaths $::env(BOARD_REPO_PATHS)
}

puts "===> root_dir   = $root_dir"
puts "===> proj_dir   = $proj_dir"
puts "===> ip_zip     = $ip_zip"
puts "===> ip_repo    = $ip_repo"

# Step 1: unzip HLS IP (export.zip)
# The Vitis HLS export creates export.zip; this script unzips it once
# into an IP repository folder that Vivado can use.
if {![file exists $ip_repo]} {
    file mkdir $ip_repo
    if {[file exists $ip_zip]} {
        puts "===> Unzipping HLS IP export.zip ..."
        exec unzip -o $ip_zip -d $ip_repo
    } else {
        puts "ERROR: HLS IP archive not found: $ip_zip"
        exit 1
    }
} else {
    puts "===> IP repository directory already exists, skipping unzip."
}

# Step 2: create Vivado project
create_project -force $proj_name $proj_dir -part $part_name

# Attach whatever ZCU104 board revision is available (version-agnostic).
set board_part [lindex [get_board_parts -quiet $board_match] 0]
if {$board_part ne ""} {
    set_property board_part $board_part [current_project]
    puts "===> Using board_part $board_part"
} else {
    puts "WARNING: no ZCU104 board part found; only device part set. Set BOARD_REPO_PATHS to your zcu104 board files."
}

# Add HLS IP repository path and refresh IP catalog
set_property ip_repo_paths $ip_repo [current_project]
update_ip_catalog

# Step 3: create Block Design
create_bd_design $bd_name

# 3.1 Add Zynq UltraScale+ MPSoC (PS). Resolve the IP version from the catalog
# instead of pinning it, so the same script works across Vivado versions.
set ps_vlnv [lindex [lsort -dictionary [get_ipdefs -all -filter {NAME == zynq_ultra_ps_e}]] end]
create_bd_cell -type ip -vlnv $ps_vlnv $ps_name

# Apply the ZCU104 board preset (same as "Run Block Automation" for PS)
apply_bd_automation -rule xilinx.com:bd_rule:zynq_ultra_ps_e \
    -config {apply_board_preset "1"} \
    [get_bd_cells $ps_name]

# Configure DDR preset and enable S_AXI_GP0
# These property names are taken from the Tcl console of a GUI session.
set_property -dict [list \
    CONFIG.SUBPRESET1            {DDR4_MICRON_MT40A256M16GE_083E} \
    CONFIG.PSU__USE__S_AXI_GP0   {1} \
] [get_bd_cells $ps_name]

# 3.2 Add the HLS "inference" IP
create_bd_cell -type ip -vlnv $hls_ip_vlnv $hls_ip_name

# 3.3 Add AXI GPIO for the 4-bit LED output (IP version resolved from catalog)
set gpio_vlnv [lindex [lsort -dictionary [get_ipdefs -all -filter {NAME == axi_gpio}]] end]
create_bd_cell -type ip -vlnv $gpio_vlnv $gpio_name

# Configure GPIO width and bind the board interface
set_property -dict [list \
    CONFIG.C_GPIO_WIDTH         {4} \
    CONFIG.GPIO_BOARD_INTERFACE {led_4bits} \
    CONFIG.C_ALL_OUTPUTS        {1} \
] [get_bd_cells $gpio_name]

# Step 4: board and AXI connection automation

# 4.1 Connect the GPIO IP to the LED board interface
# This is equivalent to "Run Connection Automation" for the LED board interface.
apply_bd_automation -rule xilinx.com:bd_rule:board \
    -config { Board_Interface {led_4bits ( LED ) } Manual_Source {Auto} } \
    [get_bd_intf_pins axi_gpio_0/GPIO]

# 4.2 Connect AXI for GPIO: M_AXI_HPM0_FPD -> axi_gpio_0/S_AXI
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

# 4.3 Connect AXI for inference control: M_AXI_HPM0_FPD -> inference_0/s_axi_control
apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config { \
    Clk_master {Auto} \
    Clk_slave  {Auto} \
    Clk_xbar   {Auto} \
    Master     {/zynq_ultra_ps_e_0/M_AXI_HPM0_FPD} \
    Slave      {/inference_0/s_axi_control} \
    ddr_seg    {Auto} \
    intc_ip    {New AXI Interconnect} \
    master_apm {0} \
} [get_bd_intf_pins inference_0/s_axi_control]

# 4.4 Connect AXI for inference input data: inference_0/m_axi_input_r -> S_AXI_HPC0_FPD
# Note: This call is symmetric with what the GUI produced; the object is the PS port.
apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config { \
    Clk_master {Auto} \
    Clk_slave  {Auto} \
    Clk_xbar   {Auto} \
    Master     {/inference_0/m_axi_input_r} \
    Slave      {/zynq_ultra_ps_e_0/S_AXI_HPC0_FPD} \
    ddr_seg    {Auto} \
    intc_ip    {New AXI SmartConnect} \
    master_apm {0} \
} [get_bd_intf_pins zynq_ultra_ps_e_0/S_AXI_HPC0_FPD]

# 4.5 Connect AXI for inference output data: inference_0/m_axi_output_r -> S_AXI_HPC0_FPD
apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config { \
    Clk_master {/zynq_ultra_ps_e_0/pl_clk0 (100 MHz)} \
    Clk_slave  {/zynq_ultra_ps_e_0/pl_clk0 (100 MHz)} \
    Clk_xbar   {/zynq_ultra_ps_e_0/pl_clk0 (100 MHz)} \
    Master     {/inference_0/m_axi_output_r} \
    Slave      {/zynq_ultra_ps_e_0/S_AXI_HPC0_FPD} \
    ddr_seg    {Auto} \
    intc_ip    {/axi_smc} \
    master_apm {0} \
} [get_bd_intf_pins inference_0/m_axi_output_r]

# 4.6 Optional: extra AXI automation from GUI log
#     M_AXI_HPM1_FPD -> axi_gpio_0/S_AXI through ps8_0_axi_periph
#     This may generate a loop warning but matches the GUI behaviour.
apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config { \
    Clk_master {Auto} \
    Clk_slave  {/zynq_ultra_ps_e_0/pl_clk0 (100 MHz)} \
    Clk_xbar   {/zynq_ultra_ps_e_0/pl_clk0 (100 MHz)} \
    Master     {/zynq_ultra_ps_e_0/M_AXI_HPM1_FPD} \
    Slave      {/axi_gpio_0/S_AXI} \
    ddr_seg    {Auto} \
    intc_ip    {/ps8_0_axi_periph} \
    master_apm {0} \
} [get_bd_intf_pins zynq_ultra_ps_e_0/M_AXI_HPM1_FPD]

# 4.7 Connect clock and reset to the HLS IP, but only if not already connected
if {[llength [get_bd_pins $hls_ip_name/ap_clk]]} {
    # Check whether ap_clk already has a net; if not, connect it.
    set ap_clk_nets [get_bd_nets -of_objects [get_bd_pins $hls_ip_name/ap_clk]]
    if {[llength $ap_clk_nets] == 0} {
        connect_bd_net [get_bd_pins $ps_name/pl_clk0] [get_bd_pins $hls_ip_name/ap_clk]
    } else {
        puts "===> ap_clk of $hls_ip_name is already connected, skipping clock connection."
    }
}

if {[llength [get_bd_pins $hls_ip_name/ap_rst_n]]} {
    # Check whether ap_rst_n already has a net; if not, connect it.
    set ap_rst_n_nets [get_bd_nets -of_objects [get_bd_pins $hls_ip_name/ap_rst_n]]
    if {[llength $ap_rst_n_nets] == 0} {
        connect_bd_net [get_bd_pins $ps_name/pl_resetn0] [get_bd_pins $hls_ip_name/ap_rst_n]
    } else {
        puts "===> ap_rst_n of $hls_ip_name is already connected, skipping reset connection."
    }
}

# Optional: regenerate BD layout for a clean diagram (not required for builds)
regenerate_bd_layout

# Step 5: validate and save Block Design
validate_bd_design
save_bd_design

# Step 6: generate HDL wrapper
set bd_file [get_files "$proj_dir/$proj_name.srcs/sources_1/bd/$bd_name/$bd_name.bd"]

make_wrapper -files $bd_file -top

set wrapper_file "$proj_dir/$proj_name.srcs/sources_1/bd/$bd_name/hdl/${bd_name}_wrapper.v"
add_files -norecurse $wrapper_file
set_property top ${bd_name}_wrapper [current_fileset]

update_compile_order -fileset sources_1

# Step 7: run synthesis, implementation, bitstream
# Parallelize OOC IP synthesis across cores. This tcl runs on the build
# machine, so nproc reads its actual core count at run time; cap at 8 since
# the block design has only ~10 OOC runs and more jobs stop helping.
if {[catch {exec nproc} n_cpu] || ![string is integer -strict $n_cpu]} {
    set n_cpu 8
}
set n_jobs [expr {$n_cpu < 16 ? $n_cpu : 16}]
set_param general.maxThreads 16

launch_runs synth_1 -jobs $n_jobs
wait_on_run synth_1

launch_runs impl_1 -to_step write_bitstream -jobs $n_jobs
wait_on_run impl_1

# Step 8: export hardware platform (XSA with bitstream)
set xsa_file "$proj_dir/${user_proj_name}_${board_tag}.xsa"
write_hw_platform -fixed -force -include_bit -file $xsa_file

puts "===> Vivado build completed."
puts "     Project directory: $proj_dir"
puts "     Bitstream:         $proj_dir/$proj_name.runs/impl_1/${bd_name}_wrapper.bit"
puts "     XSA:               $xsa_file"

exit
