# Central Execution Service

The plan is to have this as the central executor for all strategies. All trades will be sent as a packet to this, which will hold a websocket connection with IBKR and execute. MVP will be to just place the order, but further enhancement could be to use strategies to improve execution. 

This may perhaps be where the front end lives as well
