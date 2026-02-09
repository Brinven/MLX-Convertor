# MLX Model Converter & Tester

## Project Overview
Build a GUI application for converting Hugging Face models to Apple MLX format with built-in testing capabilities. The app should be user-friendly, visually polished, and suitable for open-source release.

## Target Platform
- **Primary:** macOS with Apple Silicon (M1/M2/M3/M4)
- **Note:** MLX is Apple Silicon-only; this will not work on Windows/Linux

## Technical Stack
- **Framework:** Gradio (web-based UI, easiest to start with)
- **Backend:** mlx-lm for model conversion and inference
- **Language:** Python 3.9+
- **Alternative:** CustomTkinter for native Mac app (Phase 2 if desired)

## Core Features

### 1. Model Conversion Tab
- Input field for Hugging Face model path (e.g., "LiquidAI/LFM2-1.2B-RAG")
- Output name field for the converted model
- Quantization options:
  - 4-bit (recommended for iOS/mobile)
  - 8-bit (balanced)
  - bf16 (no quantization, full precision)
- Progress indicator during conversion
- Clear status messages (success/error)
- Ability to cancel conversion if needed

### 2. Model Testing Tab
- Model path selector (browse or manual entry)
- Prompt input (multi-line text box)
- Generation parameters:
  - Max tokens slider (50-2000)
  - Temperature slider (0-1.0)
  - Top-p slider (optional, 0-1.0)
  - Repetition penalty (optional, 1.0-1.5)
- "Generate" button
- Response display area with syntax highlighting
- Option to copy response to clipboard

### 3. Model Library/History Tab (Bonus)
- List of previously converted models
- Quick load for testing
- Display model info (size, quantization, date converted)
- Delete converted models

### 4. Settings/About Tab
- Default output directory selector
- Default quantization preference
- About section with:
  - Version info
  - MLX/mlx-lm versions
  - Link to GitHub repo
  - Credits

## File Structure
```
mlx-converter/
├── CLAUDE.md                    # This file (project instructions)
├── README.md                    # User-facing documentation
├── requirements.txt             # Python dependencies
├── app.py                       # Main Gradio application
├── src/
│   ├── __init__.py
│   ├── converter.py            # Model conversion logic
│   ├── tester.py               # Model testing/inference logic
│   └── utils.py                # Helper functions
├── models/                      # Default output directory for converted models
└── examples/
    └── example_prompts.json    # Sample prompts for testing
```

## Implementation Plan

### Phase 1: Basic Functionality (MVP)
1. **Setup**
   - Create project structure
   - Set up requirements.txt with: gradio, mlx-lm
   - Create basic README

2. **Converter Module** (`src/converter.py`)
   - Function to validate HuggingFace model path
   - Function to convert model with progress tracking
   - Error handling for common issues (network, disk space, invalid model)
   - Return conversion results (success/failure, output path, file size)

3. **Tester Module** (`src/tester.py`)
   - Function to load MLX model
   - Function to generate text with configurable parameters
   - Handle model loading errors gracefully
   - Support for streaming output (future enhancement)

4. **Main App** (`app.py`)
   - Create Gradio interface with Convert and Test tabs
   - Wire up converter module to Convert tab
   - Wire up tester module to Test tab
   - Add basic styling/theming

5. **Testing**
   - Test conversion with LFM2-1.2B-RAG
   - Test with another small model (e.g., Qwen2.5-0.5B-Instruct)
   - Verify error handling works

### Phase 2: Polish & Enhancement
1. **UI Improvements**
   - Add custom CSS for better appearance
   - Add app icon/logo
   - Improve layout spacing
   - Add tooltips for settings

2. **Model Library**
   - Implement model history tracking (JSON file)
   - Add model browser/selector
   - Display model metadata

3. **Advanced Features**
   - Batch conversion queue
   - Model presets (common models with recommended settings)
   - Export/share conversion configs
   - Optional upload to HuggingFace Hub

4. **Documentation**
   - Comprehensive README with screenshots
   - Example usage guide
   - Troubleshooting section
   - Video demo (optional)

### Phase 3: Distribution (Optional)
1. **Package as Standalone App**
   - Consider py2app for Mac .app bundle
   - Or keep as Python script with easy install instructions

2. **Open Source Release**
   - Clean up code with proper comments
   - Add LICENSE (MIT recommended)
   - Create release on GitHub
   - Add contribution guidelines

## Key Requirements

### User Experience
- **Simple:** Should be usable by non-technical users
- **Fast:** Conversions complete in under 5 minutes for small models
- **Clear:** All buttons, fields, and messages should be self-explanatory
- **Safe:** Confirm before overwriting existing models
- **Helpful:** Provide good error messages with solutions

### Code Quality
- **Modular:** Keep conversion, testing, and UI logic separated
- **Documented:** Docstrings for all functions
- **Error Handling:** Graceful failures with user-friendly messages
- **Tested:** Verify core functionality works

### Technical Constraints
- **Mac Only:** Prominently note Apple Silicon requirement
- **Dependencies:** Keep minimal (Gradio + mlx-lm + standard library)
- **Storage:** Models can be large; allow custom output directory
- **Memory:** Respect system resources during conversion

## Example Prompts for Testing

Include these in `examples/example_prompts.json`:
```json
{
  "simple": "What is the capital of France?",
  "rag": "Use the following context to answer the question:\n\nContext: Paris is the capital and largest city of France.\n\nQuestion: What is the capital of France?",
  "creative": "Write a haiku about programming.",
  "technical": "Explain what a neural network is in simple terms.",
  "multilingual": "Traduire en français: Hello, how are you?"
}
```

## Default Values
- **Quantization:** 4-bit
- **Max tokens:** 512
- **Temperature:** 0.7
- **Output directory:** `./models/`

## Error Messages to Handle
- Model not found on HuggingFace
- Invalid model architecture (not supported by MLX)
- Insufficient disk space
- Network connectivity issues
- Model path doesn't exist (for testing)
- Corrupted model files

## Success Criteria
- [ ] Can convert LFM2-1.2B-RAG successfully
- [ ] Can test the converted model with various prompts
- [ ] UI is intuitive and looks polished
- [ ] Error messages are helpful
- [ ] Code is well-organized and documented
- [ ] README explains setup and usage clearly
- [ ] Ready for open-source release

## Future Enhancements (Optional)
- Support for GGUF format conversion
- Model comparison tool (test multiple models with same prompt)
- Performance benchmarking
- Fine-tuning interface
- Chat interface with conversation history
- Integration with LM Studio or other frontends

## Notes for Claude Code
- Start with MVP (Phase 1) - get basic conversion and testing working
- Use Gradio's built-in themes for professional appearance
- Test each module independently before integration
- Add progress bars/spinners for long operations
- Use threading to prevent UI freezing during conversion
- Include example usage in README with screenshots
- Make the app visually appealing - this will be open source!

## Testing Checklist
- [ ] Conversion with valid model path
- [ ] Conversion with invalid model path
- [ ] Conversion with no internet connection
- [ ] Testing with converted model
- [ ] Testing with non-existent model
- [ ] All quantization options work
- [ ] UI responsive during long operations
- [ ] Error messages display correctly

## References
- MLX GitHub: https://github.com/ml-explore/mlx
- mlx-lm Documentation: https://github.com/ml-explore/mlx-examples/tree/main/llms
- Gradio Documentation: https://www.gradio.app/docs
- LFM2-1.2B-RAG: https://huggingface.co/LiquidAI/LFM2-1.2B-RAG
