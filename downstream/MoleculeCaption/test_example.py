from transformers import T5Tokenizer, T5ForConditionalGeneration, T5EncoderModel
from transformers.modeling_outputs import BaseModelOutput
import torch

# tokenizer = T5Tokenizer.from_pretrained("molt5-"+"small", model_max_length=512)
# model = T5ForConditionalGeneration.from_pretrained("molt5-"+"small").cuda()
tokenizer = T5Tokenizer.from_pretrained("molt5-"+"small"+"-smiles2caption/", model_max_length=512)
model = T5ForConditionalGeneration.from_pretrained("molt5-"+"small"+"-smiles2caption/").cuda()

input_text = [
    'CCCCCCCCCCCCCCCCCCCCCCCCCC(=O)N[C@@H](CO[C@@H]1[C@@H]([C@H]([C@H]([C@H](O1)CO)OCC2=CC=C(C=C2)F)O)O)[C@@H]([C@@H](CCCCCCCCCCCCCC)O)O',
    'CC(=O)N[C@@H]1[C@H](C[C@](O[C@H]1[C@@H]([C@@H](CO)O)O)(C(=O)O)OC[C@@H]2[C@@H]([C@@H]([C@H]([C@H](O2)O)NC(=O)C)O[C@H]3[C@@H]([C@H]([C@H]([C@H](O3)CO)O)O)O)O)O',
    'CCCCCCCCCCCCC1=CC=C(C=C1)S(=O)(=O)O',
    ]
gt = [
    'The molecule is a glycophytoceramide having a 4-O-(4-fluorobenzyl)-alpha-D-galactosyl residue at the O-1 position and a hexacosanoyl group attached to the nitrogen. One of a series of an extensive set of 4"-O-alkylated alpha-GalCer analogues evaluated (PMID:30556652) as invariant natural killer T-cell (iNKT) antigens. It derives from an alpha-D-galactose.',
    'The molecule is a branched amino trisaccharide that consists of N-acetyl-alpha-D-galactosamine having a beta-D-galactosyl residue attached at the 3-position and a beta-N-acetylneuraminosyl residue attached at the 6-position. It has a role as an epitope. It is an amino trisaccharide and a galactosamine oligosaccharide.',
    'The molecule is a member of the class dodecylbenzenesulfonic acids that is benzenesulfonic acid in which the hydrogen at position 4 of the phenyl ring is substituted by a dodecyl group.',
]

input_ids_ = tokenizer(input_text, padding=True, truncation=True, return_tensors="pt")
input_ids = input_ids_['input_ids'].cuda()
encoder_attention_mask = input_ids_['attention_mask'].cuda()

text_tokens_ = tokenizer(gt, padding=True, truncation=True, return_tensors="pt")
text_mask = text_tokens_['attention_mask'].cuda()  # caption mask, decoder input mask
label = text_tokens_['input_ids'].cuda()  # caption
label = label.masked_fill(~text_mask.bool(), -100)

print(input_ids)
print(encoder_attention_mask)
print(label)
print(text_mask)



smiles_embeds = model.encoder(input_ids=input_ids, attention_mask=encoder_attention_mask).last_hidden_state

graph_rep = torch.zeros(3,512).cuda()
smiles_embeds = torch.cat([graph_rep.unsqueeze(1), smiles_embeds], dim=1)
encoder_attention_mask = torch.cat([torch.ones(3, 1).cuda(), encoder_attention_mask], dim=1)


encoder_outputs = BaseModelOutput(
        last_hidden_state=smiles_embeds,
        hidden_states=None,
        attentions=None,
    )

outputs = model.generate(
    # input_ids,
    encoder_outputs = encoder_outputs,  
    attention_mask = encoder_attention_mask,  # important
    num_beams=5,
    max_length=512,
    # decoder_input_ids=tokenizer(B*'The molecule is'),
    )
print(tokenizer.batch_decode(outputs, skip_special_tokens=True))


# model.train()
loss = model(
    encoder_outputs = encoder_outputs,
    attention_mask = encoder_attention_mask,
    decoder_attention_mask = text_mask,
    labels=label
    ).loss

print(loss.item())
    
