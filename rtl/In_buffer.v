`timescale 1ns / 1ps
`define IDLE 2'b00
`define IN_DATA 2'b01
`define CAL 2'b11
// reg [P_ARRAY_SIZE * P_DATA_WIDTH - 1:0] r_feature_buffer_mem [Feature_Width_Block_num * Feature_Length : 0];
// reg [P_ARRAY_SIZE * P_DATA_WIDTH - 1:0] r_weight_buffer_mem [P_ARRAY_SIZE * Weight_Width_Block_num * Feature_Width_Block_num : 0];
module InputBuffer
#(
    parameter integer                               P_ARRAY_SIZE = 32,
    parameter integer                               P_DATA_WIDTH = 8,
    parameter integer                               P_WEIGHT_BUFFER_DEPTH = 2400, 
    parameter integer                               P_FEATURE_BUFFER_DEPTH = 2400,
    parameter integer                               P_ROW_COUNT_WIDTH = 10,
    parameter integer                               P_K_BLOCK_COUNT_WIDTH = 5,
    parameter integer                               P_N_BLOCK_COUNT_WIDTH = 8
)(
    input                                           i_clk,
    input                                           i_rst_n,
    input gemv_mode,
    
    input [P_N_BLOCK_COUNT_WIDTH-1:0]             i_cfg_n_block_count, //1 ~ block_num


    input [P_K_BLOCK_COUNT_WIDTH-1:0]             i_cfg_k_block_count, //1 ~ block_num
    input [P_ROW_COUNT_WIDTH-1:0]                      i_cfg_row_count, //1 ~ block_num *P_ARRAY_SIZE

    input                                           i_compute_partial_last,

    input                                           i_feature_valid,
    input                                           i_feature_last,
    output                                          o_feature_ready,
    input [P_ARRAY_SIZE * P_DATA_WIDTH - 1:0]               i_feature_data,

    input                                           i_weight_valid,
    input                                           i_weight_last,
    output                                          o_weight_ready,
    input [P_ARRAY_SIZE * P_DATA_WIDTH - 1:0]               i_weight_data,

    output wire                                      o_buffer_feature_valid,
    output                                          o_buffer_feature_last,
    input                                           i_buffer_feature_ready,
    output wire [P_ARRAY_SIZE * P_DATA_WIDTH - 1:0]              o_buffer_feature_data,

    output wire                                      o_buffer_weight_valid,
    output                                          o_buffer_weight_last,
    input                                           i_buffer_weight_ready,
    output wire [P_ARRAY_SIZE * P_DATA_WIDTH - 1:0]              o_buffer_weight_data

);

// GEMV accepts one activation vector and 32*Nblocks weight words per K block.
// The feeder owns tile storage and backpressure; no duplicate GEMV RAM is needed.
wire legacy_o_feature_ready, legacy_o_weight_ready;
reg legacy_o_buffer_feature_valid, legacy_o_buffer_weight_valid;
wire legacy_o_buffer_feature_last, legacy_o_buffer_weight_last;
reg [P_ARRAY_SIZE*P_DATA_WIDTH-1:0] legacy_o_buffer_feature_data, legacy_o_buffer_weight_data;
reg r_gemv_feature_sent;
reg [P_N_BLOCK_COUNT_WIDTH+$clog2(P_ARRAY_SIZE):0] r_gemv_weight_words;
wire [P_N_BLOCK_COUNT_WIDTH+$clog2(P_ARRAY_SIZE):0] w_gemv_weight_words = i_cfg_n_block_count * P_ARRAY_SIZE;
wire w_gemv_enabled = (i_cfg_k_block_count != 0) && (i_cfg_n_block_count != 0);
assign o_feature_ready = gemv_mode ? (w_gemv_enabled && !r_gemv_feature_sent && i_buffer_feature_ready) : legacy_o_feature_ready;
assign o_weight_ready = gemv_mode ? (w_gemv_enabled && r_gemv_weight_words < w_gemv_weight_words && i_buffer_weight_ready) : legacy_o_weight_ready;
assign o_buffer_feature_valid = gemv_mode ? (w_gemv_enabled && !r_gemv_feature_sent && i_feature_valid) : legacy_o_buffer_feature_valid;
assign o_buffer_feature_data = gemv_mode ? i_feature_data : legacy_o_buffer_feature_data;
assign o_buffer_feature_last = gemv_mode ? 1'b1 : legacy_o_buffer_feature_last;
assign o_buffer_weight_valid = gemv_mode ? (w_gemv_enabled && r_gemv_weight_words < w_gemv_weight_words && i_weight_valid) : legacy_o_buffer_weight_valid;
assign o_buffer_weight_data = gemv_mode ? i_weight_data : legacy_o_buffer_weight_data;
assign o_buffer_weight_last = gemv_mode ? (r_gemv_weight_words == w_gemv_weight_words-1) : legacy_o_buffer_weight_last;
always @(posedge i_clk or negedge i_rst_n) begin
    if (!i_rst_n) begin
        r_gemv_feature_sent <= 0;
        r_gemv_weight_words <= 0;
    end else if (!gemv_mode || i_compute_partial_last) begin
        r_gemv_feature_sent <= 0;
        r_gemv_weight_words <= 0;
    end else begin
        if (i_feature_valid && o_feature_ready) r_gemv_feature_sent <= 1;
        if (i_weight_valid && o_weight_ready) r_gemv_weight_words <= r_gemv_weight_words + 1;
    end
end

localparam integer LP_F_ADDR_WIDTH = (P_FEATURE_BUFFER_DEPTH <= 1) ? 1 : $clog2(P_FEATURE_BUFFER_DEPTH);
localparam integer LP_W_ADDR_WIDTH = (P_WEIGHT_BUFFER_DEPTH <= 1) ? 1 : $clog2(P_WEIGHT_BUFFER_DEPTH);
localparam integer LP_LOG_A_SIZE = (P_ARRAY_SIZE <= 1) ? 1 : $clog2(P_ARRAY_SIZE);

wire w_start_readout;
wire w_accept_feature_word;
wire w_accept_weight_word;
wire w_accept_buffer_feature_word;
wire w_accept_buffer_weight_word;

reg [P_ARRAY_SIZE * P_DATA_WIDTH - 1:0] r_feature_buffer_mem [P_FEATURE_BUFFER_DEPTH - 1 : 0];
reg [P_ARRAY_SIZE * P_DATA_WIDTH - 1:0] r_weight_buffer_mem [P_WEIGHT_BUFFER_DEPTH - 1 : 0];
 
reg [1:0]                                    state;
reg [LP_W_ADDR_WIDTH-1:0]                    r_weight_words_expected;
reg [LP_F_ADDR_WIDTH-1:0]                    r_feature_words_expected; 

reg [LP_F_ADDR_WIDTH-1:0]                    r_feature_write_count;
reg [LP_F_ADDR_WIDTH-1:0]                    r_feature_write_addr;

reg [LP_W_ADDR_WIDTH-1:0]                    r_weight_write_count;
reg [LP_W_ADDR_WIDTH-1:0]                    r_weight_write_addr;

reg [LP_F_ADDR_WIDTH-1:0]                    r_buffer_feature_count;
reg [P_ROW_COUNT_WIDTH-1:0]                     r_feature_read_row;
reg [P_K_BLOCK_COUNT_WIDTH-1:0]            r_feature_read_k_block;
wire [LP_F_ADDR_WIDTH-1:0]                   w_feature_read_addr;

reg [LP_W_ADDR_WIDTH-1:0]                    r_buffer_weight_count;
reg [LP_W_ADDR_WIDTH-1:0]                    r_weight_read_addr;


assign legacy_o_feature_ready = (r_feature_write_count < r_feature_words_expected) & (state == `IN_DATA);
assign legacy_o_weight_ready = (r_weight_write_count < r_weight_words_expected) & (state == `IN_DATA);
assign w_feature_read_addr = r_feature_read_row * i_cfg_k_block_count + r_feature_read_k_block;
assign w_start_readout = ((((r_feature_words_expected != 0) && (r_weight_words_expected != 0) &&
                      (r_feature_write_count == r_feature_words_expected) && (r_weight_write_count == r_weight_words_expected))
                    | (i_compute_partial_last & !gemv_mode))
                    & (i_cfg_k_block_count != 0)
                    & (r_feature_read_k_block!=i_cfg_k_block_count));
assign w_accept_feature_word = !gemv_mode & i_feature_valid & legacy_o_feature_ready;
assign w_accept_weight_word = !gemv_mode & i_weight_valid & legacy_o_weight_ready;
assign w_accept_buffer_feature_word = legacy_o_buffer_feature_valid & i_buffer_feature_ready;
assign w_accept_buffer_weight_word = legacy_o_buffer_weight_valid & i_buffer_weight_ready;
assign legacy_o_buffer_feature_last = r_buffer_feature_count == i_cfg_row_count - 1;
assign legacy_o_buffer_weight_last = r_buffer_weight_count == i_cfg_n_block_count * P_ARRAY_SIZE - 1; 
always @(posedge i_clk or negedge i_rst_n) begin
    if(~i_rst_n)
        r_feature_words_expected <= 0;
    else
        r_feature_words_expected <= i_cfg_row_count * i_cfg_k_block_count;
end

reg [LP_LOG_A_SIZE + P_K_BLOCK_COUNT_WIDTH-1:0] r_k_elements_per_n_block;
always @(posedge i_clk or negedge i_rst_n) begin
    if(~i_rst_n) begin
        r_k_elements_per_n_block <= 0;
        r_weight_words_expected <= 0;
    end
    else begin
        r_k_elements_per_n_block <= i_cfg_k_block_count * P_ARRAY_SIZE;
        r_weight_words_expected <= i_cfg_n_block_count * r_k_elements_per_n_block;
    end
end

always @(posedge i_clk ) begin
    if(w_accept_feature_word)
        r_feature_buffer_mem[r_feature_write_addr] <= i_feature_data;
end

always @(posedge i_clk ) begin
    if(w_accept_weight_word)
        r_weight_buffer_mem[r_weight_write_addr] <= i_weight_data;
end

always @(posedge i_clk) begin
    legacy_o_buffer_feature_data <= r_feature_buffer_mem[w_feature_read_addr];
end

always @(posedge i_clk) begin
    legacy_o_buffer_weight_data <= r_weight_buffer_mem[r_weight_read_addr];
end

always @(posedge i_clk  or negedge i_rst_n) begin
    if(~i_rst_n)
        state <= `IDLE;
    else if (state == `IDLE & (!gemv_mode & (i_feature_valid | i_weight_valid)))
        state <= `IN_DATA;
    else if (state == `IN_DATA & w_start_readout)
        state <= `CAL;
    else if (state == `CAL & i_compute_partial_last 
            & ( r_feature_read_k_block == i_cfg_k_block_count))
        state <= `IDLE;
    else
        state <= state;
end

always @(posedge i_clk  or negedge i_rst_n) begin
    if(~i_rst_n)
        r_feature_write_count<=0;
    else if (w_start_readout)
        r_feature_write_count <= 0;
    else if (w_accept_feature_word)
        r_feature_write_count <= r_feature_write_count + 1;
    else
        r_feature_write_count <= r_feature_write_count;
end

always @(posedge i_clk  or negedge i_rst_n) begin
    if(~i_rst_n)
        r_feature_write_addr<=0;
    else if (w_accept_feature_word & i_feature_last)
        r_feature_write_addr <= 0;
    else if (w_accept_feature_word)
        r_feature_write_addr <= r_feature_write_addr + 1;
    else
        r_feature_write_addr <= r_feature_write_addr;
end

always @(posedge i_clk  or negedge i_rst_n) begin
    if(~i_rst_n)
        r_weight_write_count<=0;
    else if (w_start_readout)
        r_weight_write_count <= 0;
    else if (w_accept_weight_word)
        r_weight_write_count <= r_weight_write_count + 1;
    else
        r_weight_write_count <= r_weight_write_count;
end

always @(posedge i_clk  or negedge i_rst_n) begin
    if(~i_rst_n)
        r_weight_write_addr<=0;
    else if (w_accept_weight_word & i_weight_last)
        r_weight_write_addr <= 0;
    else if (w_accept_weight_word)
        r_weight_write_addr <= r_weight_write_addr + 1;
    else
        r_weight_write_addr <= r_weight_write_addr;
end

always @(posedge i_clk  or negedge i_rst_n) begin
    if(~i_rst_n)
        legacy_o_buffer_feature_valid <= 0;
    else if(w_start_readout)
        legacy_o_buffer_feature_valid <= 1;
    else if (w_accept_buffer_feature_word & legacy_o_buffer_feature_last)
        legacy_o_buffer_feature_valid <= 0;
    else 
        legacy_o_buffer_feature_valid<=legacy_o_buffer_feature_valid;
end

always @(posedge i_clk  or negedge i_rst_n) begin
    if(~i_rst_n)
        r_buffer_feature_count<=0;
    else if (r_buffer_feature_count == i_cfg_row_count)
        r_buffer_feature_count<=0;
    else if (w_accept_buffer_feature_word)
        r_buffer_feature_count<=r_buffer_feature_count+1;
    else
        r_buffer_feature_count <= r_buffer_feature_count;
end

always @(posedge i_clk  or negedge i_rst_n) begin
    if(~i_rst_n)
        r_feature_read_row <= 0;
    else if (w_start_readout)
        r_feature_read_row <= 1;
    else if (r_feature_read_row == i_cfg_row_count - 1)
        r_feature_read_row <= 0;
    else if (r_feature_read_row != 0 & w_accept_buffer_feature_word)
        r_feature_read_row <= r_feature_read_row + 1;
    else
        r_feature_read_row <= r_feature_read_row;
end

always @(posedge i_clk  or negedge i_rst_n) begin
    if(~i_rst_n)
       r_feature_read_k_block <= 0;
    else if (i_compute_partial_last & r_feature_read_k_block == i_cfg_k_block_count)
        r_feature_read_k_block <= 0; 
    else if(w_accept_buffer_feature_word & legacy_o_buffer_feature_last)
        r_feature_read_k_block <= r_feature_read_k_block + 1;
    else
        r_feature_read_k_block <= r_feature_read_k_block;
end

always @(posedge i_clk  or negedge i_rst_n) begin
    if(~i_rst_n)
        legacy_o_buffer_weight_valid <= 0;
    else if(w_start_readout)
        legacy_o_buffer_weight_valid <= 1;
    else if (w_accept_buffer_weight_word & legacy_o_buffer_weight_last)
        legacy_o_buffer_weight_valid <= 0;
    else 
        legacy_o_buffer_weight_valid<=legacy_o_buffer_weight_valid;
end

always @(posedge i_clk or negedge i_rst_n) begin
    if(~i_rst_n)
        r_buffer_weight_count <= 0;
    else if (r_buffer_weight_count == i_cfg_n_block_count * P_ARRAY_SIZE)
        r_buffer_weight_count <= 0;
    else if (w_accept_buffer_weight_word)
        r_buffer_weight_count<=r_buffer_weight_count+1;
    else
        r_buffer_weight_count<=r_buffer_weight_count;
end

always @(posedge i_clk or negedge i_rst_n ) begin
    if(~i_rst_n)
        r_weight_read_addr <= 0;
    else if (r_weight_read_addr== r_weight_words_expected -1)
        r_weight_read_addr <= 0;
    else if ((w_start_readout
            | w_accept_buffer_weight_word)
            & (~legacy_o_buffer_weight_last))
        r_weight_read_addr<=r_weight_read_addr+1;
    else
        r_weight_read_addr<=r_weight_read_addr;
end
endmodule
