module config_reg #(
    parameter  DATA_WIDTH= 32,
    parameter  ADDR_WIDTH = 5,
    parameter  DEPTH = 24
    
) (
    input   wire                    clk     ,
    input   wire                    rst_n   ,
    input   wire                    wr_en      ,
    input                           cs_n,
    input   [ DATA_WIDTH    -1 : 0] wr_data,
    output reg [ DATA_WIDTH -1 : 0] rd_data,
    output  wire                    resp      ,      
    input   [ ADDR_WIDTH    -1 : 0] addr                                 
);
//==============================================================================
// Constant Definition :
//==============================================================================


reg [ DATA_WIDTH    - 1 : ] mem[ 0 : DEPTH  -1];

wire wr_ready;
reg rd_valid;
//==============================================================================
// Variable Definition :
//==============================================================================
always @ ( posedge clk or negedge rst_n ) begin
    if ( !wr_en && !cs_n ) begin
        rd_data <= mem[addr];
    end
end

always @ ( posedge clk or negedge rst_n ) begin
    if ( wr_en && !cs_n ) begin
       mem[addr]  <= wr_data;
    end
end

assign wr_ready = addr < DEPTH;

always @ ( posedge clk or negedge rst_n ) begin
    if ( !rst_n ) begin
        rd_valid <= 0;
    end else if ( !wr_en && !cs_n ) begin
        rd_valid <= 1;
    end else if ( resp ) begin
        rd_valid <= 0;
    end
end
assign resp =  wr_en? wr_ready : rd_valid;


//==============================================================================
// Logic Design :
//==============================================================================
assign 





//==============================================================================
// Sub-Module :
//==============================================================================



endmodule