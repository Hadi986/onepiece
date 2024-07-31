
## Uranus API
-crop_common_resize
```console
magik::uranus::BsCommonParam param;
param.pad_val = 114;
param.pad_type = magik::uranus::BsPadType::SYMMETRY;
param.in_layout = magik::uranus::ChannelLayout::RGBA;
param.out_layout = magik::uranus::ChannelLayout::RGBA;
param.input_height = in_h;
param.input_width = in_w;
param.input_line_stride = in_w * 4;
param.addr_attr.vir_addr = in_tensor_data;

vector<uranus::Tensor> output_tesnor;
output_tesnor.push_back(out);
std::cout<<"in_tensor.shape"<<in_tensor.shape()[2]<<std::endl;
uranus::crop_common_resize(output_tesnor, output_tmp, magik::uranus::AddressLocate::NMEM_VIRTUAL, &param);
```
-uranus::memcopy()









