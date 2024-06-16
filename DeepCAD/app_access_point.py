def generate_deepcad(agent, file_path, output_path = None):
    agent.cfg.pc_root = file_path
    print("data path:", agent.cfg.pc_root)
    if output_path:
        agent.cfg.output = output_path
    agent.cfg.expPNG = True
    agent.cfg.expSTEP = True
    out_path = agent.pc2cad()
    return out_path
