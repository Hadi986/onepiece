        m = magik_quantizer.quant(m)
        ck = torch.load(opt.checkpoint)
        new_state_dict = ck.state_dict()
        for name, module in m.named_modules():
            if hasattr(module, "index") and "BinOp" in name:        
                #print(name, " >>> ", module.index)
                for (name1, module1) in ck.named_modules():
                    if hasattr(module1, "index") and ('add' in name1 or 'cat' in name1):
                        #print(name1, " --- ", module1.index)
                        if module.index == module1.index:
                            n = name+".quantize_feature.clip_min_value"
                            n1 = name1+".quantize_feature.clip_min_value"
                            new_state_dict[n] = new_state_dict[n1]
                            del new_state_dict[n1]
                            n = name+".quantize_feature.clip_max_value"
                            n1 = name1+".quantize_feature.clip_max_value"
                            new_state_dict[n] = new_state_dict[n1]
                            del new_state_dict[n1]
                            
        m.load_state_dict(new_state_dict)