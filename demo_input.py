#!/usr/bin/env python3
"""
Demo script showcasing the enhanced input features.
"""

import sys
sys.path.append('.')

from agent import InterruptibleInput

def demo_enhanced_input():
    """Demo the enhanced input features."""
    print("🚀 Enhanced Interactive Input Demo")
    print("=" * 50)
    print()
    print("📝 New Features:")
    print("   • Arrow keys: ← → for cursor movement")
    print("   • Home/End: Move to start/end of line")
    print("   • Ctrl+A/E: Alternative Home/End")
    print("   • Ctrl+K: Kill text from cursor to end")
    print("   • Ctrl+U: Kill entire line")
    print("   • Delete key: Delete character at cursor")
    print("   • Smart scrolling: Long lines scroll smoothly")
    print("   • Paste detection: Multiline content handled automatically")
    print("   • ESC: Interrupt/cancel input")
    print()
    print("💡 Try typing a very long line to see horizontal scrolling!")
    print("💡 Try pasting multiple lines to see paste detection!")
    print("💡 IMPORTANT: Test arrow keys - they should only move cursor, not add characters!")
    print("💡 Type 'quit' to exit")
    print()
    
    input_handler = InterruptibleInput()
    
    try:
        while True:
            print("-" * 30)
            result = input_handler.get_input("Enhanced> ")
            
            if result is None:
                print("🛑 Input was interrupted!")
                break
            elif result.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            else:
                print(f"✓ Input received: '{result}'")
                print(f"  Length: {len(result)} characters")
                if '\n' in result:
                    lines = result.split('\n')
                    print(f"  Lines: {len(lines)}")
                    for i, line in enumerate(lines[:3], 1):  # Show first 3 lines
                        print(f"    Line {i}: '{line}'")
                    if len(lines) > 3:
                        print(f"    ... and {len(lines) - 3} more lines")
                
    except KeyboardInterrupt:
        print("\n🛑 Demo interrupted by Ctrl+C!")

if __name__ == "__main__":
    demo_enhanced_input()