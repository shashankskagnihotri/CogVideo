from comfyui_glm4_wrapper import GLM4ModelLoader, GLM4PromptEnhancer, GLM4Inference

# Load the model
model_loader = GLM4ModelLoader()
pipeline = model_loader.gen(model="THUDM/glm-4v-9b", precision="bf16", quantization="8")[0]

# Enhance the prompt
enhancer = GLM4PromptEnhancer()
enhanced_prompt = enhancer.enhance_prompt(
  GLMPipeline=pipeline,
  prompt="A beautiful sunrise over the mountains",
  max_new_tokens=200,
  temperature=0.1,
  top_k=40,
  top_p=0.7,
  repetition_penalty=1.1,
  image=None,  # PIL Image
  unload_model=True,
  seed=42
)
print(enhanced_prompt)