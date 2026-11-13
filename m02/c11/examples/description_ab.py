from _shop import MODEL, ORDER_ARG, TOOLS, client, fn_tool

ID = {"id": {"type": "string"}}
VAGUE = [fn_tool("lookup", "Lookup.", ID), fn_tool("calc", "Calc.", ID), fn_tool("create", "Create.", {**ID, "text": {"type": "string"}})]
CLEAR = TOOLS + [fn_tool("create_ticket", "Open a support ticket for a human agent: refunds, complaints, address changes. "
                         "Not for status or arrival time questions.",
                         {**ORDER_ARG, "summary": {"type": "string", "description": "One sentence in Vietnamese"}})]
GOLD = {"vague": {"status": "lookup", "eta": "calc", "ticket": "create"},
        "clear": {"status": "get_order_status", "eta": "predict_eta", "ticket": "create_ticket"}}
MESSAGES = [("Đơn AT047838 đang ở đâu rồi?", "status"), ("bao lâu nữa đơn AT051122 tới", "eta"),
            ("Tôi muốn hoàn tiền đơn AT039405", "ticket"), ("don AT060017 giao chua vay", "status"),
            ("Mấy giờ thì đơn AT047838 đến nơi?", "eta"), ("Đổi địa chỉ nhận đơn AT052290 giúp tôi", "ticket")]
INSTRUCTIONS = "You are the support assistant of Giao Hang Anh Ti. Use exactly one tool to answer."

llm = client()
for label, tools in [("vague", VAGUE), ("clear", CLEAR)]:
    hits = 0
    for text, need in MESSAGES:
        r = llm.responses.create(model=MODEL, instructions=INSTRUCTIONS, input=text, tools=tools,
                                 reasoning={"effort": "none"}, max_output_tokens=300, temperature=0)
        called = [o.name for o in r.output if o.type == "function_call"]
        ok = called == [GOLD[label][need]]
        hits += ok
        print(f"{label:<6}{need:<8}{'ok ' if ok else 'BAD'} called={called or 'none, replied with text'}")
    print(f"{label}: {hits}/{len(MESSAGES)} messages got the right tool")
