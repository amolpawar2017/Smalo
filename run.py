#!/usr/bin/env python3
"""
Final SmolLM3-3B Optimized Script for Linux
Uses official HuggingFace parameters: temperature=0.6, top_p=0.95
No parameter conflicts, maximum CPU performance
"""

import os
import torch
import psutil
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
from contextlib import contextmanager

def setup_pytorch_optimized():
    """Setup PyTorch for optimal CPU performance"""
    cpu_count = psutil.cpu_count(logical=True)
    physical_cpu_count = psutil.cpu_count(logical=False)
    
    # Optimal thread configuration
    if physical_cpu_count >= 8:
        num_threads = min(physical_cpu_count, 8)
    elif physical_cpu_count >= 4:
        num_threads = physical_cpu_count
    else:
        num_threads = cpu_count
    
    torch.set_num_threads(num_threads)
    torch.set_num_interop_threads(2)
    torch.backends.mkldnn.enabled = True
    
    # Environment variables for performance
    os.environ["OMP_NUM_THREADS"] = str(num_threads)
    os.environ["MKL_NUM_THREADS"] = str(num_threads)
    os.environ["MKL_DYNAMIC"] = "FALSE"
    
    print(f"✓ PyTorch optimized: {num_threads} threads, MKL-DNN enabled")

@contextmanager
def timer():
    """Simple timer context manager"""
    start = time.time()
    yield
    end = time.time()
    print(f"⏱️  Time: {end - start:.2f}s")

class SmolLM3Generator:
    def __init__(self, model_path):
        self.model_path = model_path
        self.tokenizer = None
        self.model = None
        
    def load_model(self):
        """Load model using official HuggingFace method"""
        print("🔄 Loading tokenizer...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            print("✓ Tokenizer loaded")
        except Exception as e:
            print(f"❌ Tokenizer error: {e}")
            return False
            
        print("🔄 Loading model...")
        try:
            with timer():
                # Official loading method from HuggingFace docs
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path
                ).to("cpu")
                self.model.eval()
            print("✓ Model loaded successfully")
        except Exception as e:
            print(f"❌ Model error: {e}")
            return False
            
        return True
    
    def generate_chat(self, prompt, max_new_tokens=200, use_thinking=False):
        """Generate using official chat template method"""
        try:
            with torch.no_grad(), timer():
                # Official chat template approach
                if use_thinking:
                    messages = [{"role": "user", "content": prompt}]
                else:
                    messages = [
                        {"role": "system", "content": "/no_think"},
                        {"role": "user", "content": prompt}
                    ]
                
                # Apply chat template - official method
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                
                # Tokenize and generate
                model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
                
                # Official parameters: temperature=0.6, top_p=0.95
                generated_ids = self.model.generate(
                    **model_inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=0.6,    # Official HF recommendation
                    top_p=0.95,        # Official HF recommendation
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
                
                # Decode only new tokens
                output_ids = generated_ids[0][len(model_inputs.input_ids[0]):]
                response = self.tokenizer.decode(output_ids, skip_special_tokens=True)
                
                return response.strip()
                
        except Exception as e:
            print(f"❌ Chat generation error: {e}")
            return None
    
    def generate_simple(self, prompt, max_new_tokens=100):
        """Simple generation without chat template"""
        try:
            with torch.no_grad(), timer():
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
                
                # Official parameters
                outputs = self.model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask if 'attention_mask' in inputs else None,
                    max_new_tokens=max_new_tokens,
                    temperature=0.6,
                    top_p=0.95,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
                
                # Decode only new tokens
                new_tokens = outputs[0][inputs.input_ids.shape[1]:]
                response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
                
                return response.strip()
                
        except Exception as e:
            print(f"❌ Simple generation error: {e}")
            return None
    
    def generate_with_tools(self, prompt, tools=None):
        """Generate with tool calling - official method"""
        try:
            with torch.no_grad(), timer():
                messages = [{"role": "user", "content": prompt}]
                
                if tools:
                    # Official tool calling method
                    inputs = self.tokenizer.apply_chat_template(
                        messages,
                        enable_thinking=False,
                        xml_tools=tools,
                        add_generation_prompt=True,
                        tokenize=True,
                        return_tensors="pt"
                    ).to(self.model.device)
                    
                    outputs = self.model.generate(
                        inputs,
                        max_new_tokens=500,
                        temperature=0.6,
                        top_p=0.95,
                        do_sample=True
                    )
                    
                    response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                    return response
                else:
                    return self.generate_chat(prompt)
                    
        except Exception as e:
            print(f"❌ Tool generation error: {e}")
            return None
    
    def benchmark(self):
        """Performance benchmark with official parameters"""
        print("\n🏁 PERFORMANCE BENCHMARK")
        print("=" * 50)
        print("Using official HF parameters: temperature=0.6, top_p=0.95")
        
        test_cases = [
            ("Simple", "Hello, how are you?", "simple", 50),
            ("Chat", "Explain quantum computing briefly", "chat", 150),
            ("Thinking", "Solve: What is 15% of 240?", "thinking", 200)
        ]
        
        total_time = 0
        total_tokens = 0
        
        for name, prompt, method, max_tokens in test_cases:
            print(f"\n📊 {name} Test: '{prompt}'")
            
            start_time = time.time()
            
            if method == "simple":
                result = self.generate_simple(prompt, max_tokens)
            elif method == "thinking":
                result = self.generate_chat(prompt, max_tokens, use_thinking=True)
            else:
                result = self.generate_chat(prompt, max_tokens, use_thinking=False)
            
            end_time = time.time()
            duration = end_time - start_time
            total_time += duration
            
            if result:
                token_count = len(result.split())
                total_tokens += token_count
                tokens_per_sec = token_count / duration if duration > 0 else 0
                
                print(f"✓ Output: {result[:100]}{'...' if len(result) > 100 else ''}")
                print(f"📈 {token_count} tokens in {duration:.2f}s ({tokens_per_sec:.1f} tokens/sec)")
            else:
                print("❌ Generation failed")
        
        if total_tokens > 0:
            avg_speed = total_tokens / total_time
            print(f"\n🎯 Average Performance: {avg_speed:.1f} tokens/sec")

def main():
    print("🚀 SmolLM3-3B Final Optimized Script")
    print("=" * 50)
    print("Official HuggingFace parameters: temperature=0.6, top_p=0.95")
    
    # UPDATE THIS PATH TO YOUR MODEL LOCATION
    model_path = "/home/user/Model/smalo3"  # ← CHANGE THIS
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        print("Please update the model_path variable above")
        return
    
    print(f"📁 Model: {model_path}")
    
    # Setup optimizations
    setup_pytorch_optimized()
    
    # Load model
    generator = SmolLM3Generator(model_path)
    if not generator.load_model():
        return
    
    # Run benchmark
    generator.benchmark()
    
    # Example tools for tool calling demo
    demo_tools = [{
        "name": "get_weather",
        "description": "Get weather for a city",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"}
            }
        }
    }]
    
    # Interactive mode
    print("\n" + "=" * 50)
    print("💬 INTERACTIVE MODE")
    print("Commands:")
    print("  'quit' - Exit")
    print("  'bench' - Run benchmark")
    print("  'think: <prompt>' - Use thinking mode")
    print("  'simple: <prompt>' - Simple generation")
    print("  'tool: <prompt>' - Tool calling demo")
    print("=" * 50)
    
    while True:
        user_input = input("\n🎯 Enter prompt: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break
        elif user_input.lower() == 'bench':
            generator.benchmark()
            continue
        
        if user_input:
            if user_input.startswith('think:'):
                prompt = user_input[6:].strip()
                print("🧠 Thinking mode...")
                result = generator.generate_chat(prompt, max_new_tokens=300, use_thinking=True)
            elif user_input.startswith('simple:'):
                prompt = user_input[7:].strip()
                print("⚡ Simple generation...")
                result = generator.generate_simple(prompt, max_new_tokens=150)
            elif user_input.startswith('tool:'):
                prompt = user_input[5:].strip()
                print("🔧 Tool calling...")
                result = generator.generate_with_tools(prompt, tools=demo_tools)
            else:
                # Default: chat mode without thinking
                result = generator.generate_chat(user_input, max_new_tokens=250)
            
            if result:
                print(f"🤖 Response: {result}")
            else:
                print("❌ Generation failed")

if __name__ == "__main__":
    main()
