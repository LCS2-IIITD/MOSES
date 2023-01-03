from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

SOURCE_MAX_LEN = 480
TARGET_MAX_LEN = 50
MAX_UTTERANCES = 25

ACOUSTIC_DIM = 154
ACOUSTIC_MAX_LEN = 600

VISUAL_DIM = 2048
VISUAL_MAX_LEN = 96

TTS_DIM = 88
TTS_MAX_LEN = 1

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("Using GPU")
else:
    DEVICE = torch.device("cpu")
    print("Using CPU")

class ContextAwareAttention(nn.Module):
    def __init__(self, dim_model: int, dim_context: int):
        super(ContextAwareAttention, self).__init__()
        self.dim_model = dim_model
        self.dim_context = dim_context
        self.attention_layer = nn.MultiheadAttention(embed_dim=self.dim_model, num_heads=1, bias=True, add_zero_attn=False, batch_first=True, device=DEVICE)

        self.u_k = nn.Linear(self.dim_context, self.dim_model, bias=False)
        self.w1_k = nn.Linear(self.dim_model, 1, bias=False)
        self.w2_k = nn.Linear(self.dim_model, 1, bias=False)
        
        self.u_v = nn.Linear(self.dim_context, self.dim_model, bias=False)
        self.w1_v = nn.Linear(self.dim_model, 1, bias=False)
        self.w2_v = nn.Linear(self.dim_model, 1, bias=False)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, context: Optional[torch.Tensor]=None):
        key_context = self.u_k(context)
        value_context = self.u_v(context)

        lambda_k = F.sigmoid(self.w1_k(k) + self.w2_k(key_context))
        lambda_v = F.sigmoid(self.w1_v(v) + self.w2_v(value_context))

        k_cap = (1 - lambda_k) * k + lambda_k * key_context
        v_cap = (1 - lambda_v) * v + lambda_v * value_context

        attention_output, _ = self.attention_layer(query=q, key=k_cap, value=v_cap)
        return attention_output

class MAF_TT(nn.Module):
    def __init__(self, dim_model: int):
        super(MAF_TT, self).__init__()
        self.tts_context_transform = nn.Linear(TTS_MAX_LEN, SOURCE_MAX_LEN, bias=False)
        self.tts_context_attention = ContextAwareAttention(dim_model=dim_model, dim_context=TTS_DIM)
        self.tts_gate = nn.Linear(dim_model + dim_model, dim_model)
        self.final_layer_norm = nn.LayerNorm(dim_model)
        
    def forward(self, text_input: torch.Tensor, tts_context: Optional[torch.Tensor]=None):
        # TTS as Context for Attention
        tts_context = tts_context.permute(0, 2, 1)
        tts_context = self.tts_context_transform(tts_context)
        tts_context = tts_context.permute(0, 2, 1)
        
        tts_out = self.tts_context_attention(q=text_input, k=text_input, v=text_input, context=tts_context)
        
        # Global Information Fusion Mechanism
        weight_t = F.sigmoid(self.tts_gate(torch.cat((tts_out, text_input), dim=-1)))
        output = self.final_layer_norm(text_input + weight_t * tts_out)

        return output

class MAF_Ta(nn.Module):
    def __init__(self, dim_model: int):
        super(MAF_Ta, self).__init__()
        self.acoustic_context_transform = nn.Linear(ACOUSTIC_MAX_LEN, SOURCE_MAX_LEN, bias=False)     
        self.acoustic_context_attention = ContextAwareAttention(dim_model=dim_model, dim_context=ACOUSTIC_DIM)
        self.acoustic_gate = nn.Linear(dim_model + dim_model, dim_model)
        self.final_layer_norm = nn.LayerNorm(dim_model)
        
    def forward(self, text_input: torch.Tensor, acoustic_context: Optional[torch.Tensor]=None):    
        # Audio as Context for Attention
        acoustic_context = acoustic_context.permute(0, 2, 1)
        acoustic_context = self.acoustic_context_transform(acoustic_context)
        acoustic_context = acoustic_context.permute(0, 2, 1)

        audio_out = self.acoustic_context_attention(q=text_input, k=text_input, v=text_input, context=acoustic_context)
        
        # Global Information Fusion Mechanism
        weight_a = F.sigmoid(self.acoustic_gate(torch.cat((audio_out, text_input), dim=-1)))        
        output = self.final_layer_norm(text_input + weight_a * audio_out)

        return output
    
    
class MAF_tA(nn.Module):
    def __init__(self, dim_model: int):
        super(MAF_tA, self).__init__()
        self.acoustic_context_transform = nn.Linear(ACOUSTIC_MAX_LEN, SOURCE_MAX_LEN, bias=False)     
        self.acoustic_context_attention = ContextAwareAttention(dim_model=ACOUSTIC_DIM, dim_context=dim_model)
        self.correct_dim = nn.Linear(ACOUSTIC_DIM, dim_model)
        self.acoustic_gate = nn.Linear(dim_model + dim_model, dim_model)
        self.final_layer_norm = nn.LayerNorm(dim_model)
        
    def forward(self, text_input: torch.Tensor, acoustic_context: Optional[torch.Tensor]=None):    
        # Audio as Context for Attention
        acoustic_context = acoustic_context.permute(0, 2, 1)
        acoustic_context = self.acoustic_context_transform(acoustic_context)
        acoustic_context = acoustic_context.permute(0, 2, 1)

        audio_out = self.acoustic_context_attention(q=acoustic_context, k=acoustic_context, v=acoustic_context, context=text_input)
        audio_out = self.correct_dim(audio_out)
        
        # Global Information Fusion Mechanism
        weight_a = F.sigmoid(self.acoustic_gate(torch.cat((audio_out, text_input), dim=-1)))        
        output = self.final_layer_norm(text_input + weight_a * audio_out)

        return output
    
    
class MAF_Tv(nn.Module):
    def __init__(self, dim_model: int):
        super(MAF_Tv, self).__init__()
        self.visual_context_transform = nn.Linear(VISUAL_MAX_LEN, SOURCE_MAX_LEN, bias=False)
        self.visual_context_attention = ContextAwareAttention(dim_model=dim_model, dim_context=VISUAL_DIM)
        self.visual_gate = nn.Linear(dim_model + dim_model, dim_model)
        self.final_layer_norm = nn.LayerNorm(dim_model)
        
    def forward(self, text_input: torch.Tensor, visual_context: Optional[torch.Tensor]=None):
        # Video as Context for Attention
        visual_context = visual_context.permute(0, 2, 1)
        visual_context = self.visual_context_transform(visual_context)
        visual_context = visual_context.permute(0, 2, 1)
        
        video_out = self.visual_context_attention(q=text_input, k=text_input, v=text_input, context=visual_context)
        
        # Global Information Fusion Mechanism
        weight_v = F.sigmoid(self.visual_gate(torch.cat((video_out, text_input), dim=-1)))
        output = self.final_layer_norm(text_input + weight_v * video_out)
        
        return output
    
class MAF_tV(nn.Module):
    def __init__(self, dim_model: int):
        super(MAF_tV, self).__init__()
        self.visual_context_transform = nn.Linear(VISUAL_MAX_LEN, SOURCE_MAX_LEN, bias=False)
        self.visual_context_attention = ContextAwareAttention(dim_model=VISUAL_DIM, dim_context=dim_model)
        self.correct_dim = nn.Linear(VISUAL_DIM, dim_model)
        self.visual_gate = nn.Linear(dim_model + dim_model, dim_model)
        self.final_layer_norm = nn.LayerNorm(dim_model)
        
    def forward(self, text_input: torch.Tensor, visual_context: Optional[torch.Tensor]=None):
        # Video as Context for Attention
        visual_context = visual_context.permute(0, 2, 1)
        visual_context = self.visual_context_transform(visual_context)
        visual_context = visual_context.permute(0, 2, 1)
        
        video_out = self.visual_context_attention(q=visual_context, k=visual_context, v=visual_context, context=text_input)
        video_out = self.correct_dim(video_out)
        
        # Global Information Fusion Mechanism
        weight_v = F.sigmoid(self.visual_gate(torch.cat((video_out, text_input), dim=-1)))
        output = self.final_layer_norm(text_input + weight_v * video_out)
        
        return output
    
    
class MAF_Tav(nn.Module):
    def __init__(self, dim_model: int):
        super(MAF_Tav, self).__init__()
        self.acoustic_context_transform = nn.Linear(ACOUSTIC_MAX_LEN, SOURCE_MAX_LEN, bias=False)     
        self.visual_context_transform = nn.Linear(VISUAL_MAX_LEN, SOURCE_MAX_LEN, bias=False)
        self.acoustic_context_attention = ContextAwareAttention(dim_model=dim_model, dim_context=ACOUSTIC_DIM)
        self.visual_context_attention = ContextAwareAttention(dim_model=dim_model, dim_context=VISUAL_DIM)
        self.acoustic_gate = nn.Linear(dim_model + dim_model, dim_model)
        self.visual_gate = nn.Linear(dim_model + dim_model, dim_model)
        self.final_layer_norm = nn.LayerNorm(dim_model)
        
    def forward(self, text_input: torch.Tensor, acoustic_context: Optional[torch.Tensor]=None, visual_context: Optional[torch.Tensor]=None): 
        # Audio as Context for Attention
        acoustic_context = acoustic_context.permute(0, 2, 1)
        acoustic_context = self.acoustic_context_transform(acoustic_context)
        acoustic_context = acoustic_context.permute(0, 2, 1)
        
        audio_out = self.acoustic_context_attention(q=text_input, k=text_input, v=text_input, context=acoustic_context)
        
        # Video as Context for Attention
        visual_context = visual_context.permute(0, 2, 1)
        visual_context = self.visual_context_transform(visual_context)
        visual_context = visual_context.permute(0, 2, 1)
        
        video_out = self.visual_context_attention(q=text_input,  k=text_input, v=text_input, context=visual_context)

        # Global Information Fusion Mechanism
        weight_a = F.sigmoid(self.acoustic_gate(torch.cat((audio_out, text_input), dim=-1)))
        weight_v = F.sigmoid(self.visual_gate(torch.cat((video_out, text_input), dim=-1)))
        
        output = self.final_layer_norm(text_input + weight_a * audio_out + weight_v * video_out)

        return output
    
    
class MAF_tAv(nn.Module):
    def __init__(self, dim_model: int):
        super(MAF_tAv, self).__init__()
        self.acoustic_context_transform = nn.Linear(ACOUSTIC_MAX_LEN, SOURCE_MAX_LEN, bias=False)     
        self.visual_context_transform = nn.Linear(VISUAL_MAX_LEN, SOURCE_MAX_LEN, bias=False)
        self.acoustic_context_attention = ContextAwareAttention(dim_model=ACOUSTIC_DIM, dim_context=dim_model)
        self.visual_context_attention = ContextAwareAttention(dim_model=dim_model, dim_context=VISUAL_DIM)
        self.correct_dim = nn.Linear(ACOUSTIC_DIM, dim_model)
        self.acoustic_gate = nn.Linear(dim_model + dim_model, dim_model)
        self.visual_gate = nn.Linear(dim_model + dim_model, dim_model)
        self.final_layer_norm = nn.LayerNorm(dim_model)
        
    def forward(self, text_input: torch.Tensor, acoustic_context: Optional[torch.Tensor]=None, visual_context: Optional[torch.Tensor]=None): 
        # Audio as Context for Attention
        acoustic_context = acoustic_context.permute(0, 2, 1)
        acoustic_context = self.acoustic_context_transform(acoustic_context)
        acoustic_context = acoustic_context.permute(0, 2, 1)
        
        audio_out = self.acoustic_context_attention(q=acoustic_context, k=acoustic_context, v=acoustic_context, context=text_input)
        audio_out = self.correct_dim(audio_out)
        
        # Video as Context for Attention
        visual_context = visual_context.permute(0, 2, 1)
        visual_context = self.visual_context_transform(visual_context)
        visual_context = visual_context.permute(0, 2, 1)
        
        video_out = self.visual_context_attention(q=text_input,  k=text_input, v=text_input, context=visual_context)

        # Global Information Fusion Mechanism
        weight_a = F.sigmoid(self.acoustic_gate(torch.cat((audio_out, text_input), dim=-1)))
        weight_v = F.sigmoid(self.visual_gate(torch.cat((video_out, text_input), dim=-1)))
        
        output = self.final_layer_norm(text_input + weight_a * audio_out + weight_v * video_out)

        return output
    
    
class MAF_taV(nn.Module):
    def __init__(self, dim_model: int):
        super(MAF_taV, self).__init__()
        self.acoustic_context_transform = nn.Linear(ACOUSTIC_MAX_LEN, SOURCE_MAX_LEN, bias=False)     
        self.visual_context_transform = nn.Linear(VISUAL_MAX_LEN, SOURCE_MAX_LEN, bias=False)
        self.acoustic_context_attention = ContextAwareAttention(dim_model=dim_model, dim_context=ACOUSTIC_DIM)
        self.visual_context_attention = ContextAwareAttention(dim_model=VISUAL_DIM, dim_context=dim_model)
        self.correct_dim = nn.Linear(VISUAL_DIM, dim_model)
        self.acoustic_gate = nn.Linear(dim_model+ dim_model, dim_model)
        self.visual_gate = nn.Linear(dim_model + dim_model, dim_model)
        self.final_layer_norm = nn.LayerNorm(dim_model)
        
    def forward(self, text_input: torch.Tensor, acoustic_context: Optional[torch.Tensor]=None, visual_context: Optional[torch.Tensor]=None): 
        # Audio as Context for Attention
        acoustic_context = acoustic_context.permute(0, 2, 1)
        acoustic_context = self.acoustic_context_transform(acoustic_context)
        acoustic_context = acoustic_context.permute(0, 2, 1)
        
        audio_out = self.acoustic_context_attention(q=text_input, k=text_input, v=text_input, context=acoustic_context)
        
        # Video as Context for Attention
        visual_context = visual_context.permute(0, 2, 1)
        visual_context = self.visual_context_transform(visual_context)
        visual_context = visual_context.permute(0, 2, 1)
        
        video_out = self.visual_context_attention(q=visual_context,  k=visual_context, v=visual_context, context=text_input)
        video_out = self.correct_dim(video_out)

        # Global Information Fusion Mechanism
        weight_a = F.sigmoid(self.acoustic_gate(torch.cat((audio_out, text_input), dim=-1)))
        weight_v = F.sigmoid(self.visual_gate(torch.cat((video_out, text_input), dim=-1)))
        
        output = self.final_layer_norm(text_input + weight_a * audio_out + weight_v * video_out)

        return output