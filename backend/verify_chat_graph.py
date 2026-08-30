import json
import os
from pathlib import Path
from backend.dataset.chat_parser import ChatGPTParser

def create_synthetic_data(path: Path):
    data = [
        # SCENARIO 1: Hash Sensitivity Test
        {
            "title": "Conversation A (Original)",
            "current_node": "node2",
            "mapping": {
                "node1": {
                    "parent": None,
                    "message": {
                        "author": {"role": "user"},
                        "content": {"content_type": "text", "parts": ["Tell me a joke."]}
                    }
                },
                "node2": {
                    "parent": "node1",
                    "message": {
                        "author": {"role": "assistant"},
                        "content": {"content_type": "text", "parts": ["Why did the chicken cross the road?"]}
                    }
                }
            }
        },
        {
            "title": "Conversation A (Edited - 1 char diff)",
            "current_node": "node2",
            "mapping": {
                "node1": {
                    "parent": None,
                    "message": {
                        "author": {"role": "user"},
                        "content": {"content_type": "text", "parts": ["Tell me a joke."]}
                    }
                },
                "node2": {
                    "parent": "node1",
                    "message": {
                        "author": {"role": "assistant"},
                        "content": {"content_type": "text", "parts": ["Why did the chicken cross the road!"]}  # Changed ? to !
                    }
                }
            }
        },
        
        # SCENARIO 2: Branching and Graph Traversal Test
        # We deliberately shuffle keys out of chronological order
        # Node structure:
        # A (user) -> B (assistant) -> C1 (user, branch 1, abandoned)
        #                           -> C2 (user, branch 2, active)
        # current_node = C2
        {
            "title": "Branched Conversation",
            "current_node": "key_C2_active",
            "mapping": {
                "key_C1_abandoned": {
                    "parent": "key_B",
                    "message": {
                        "author": {"role": "user"},
                        "content": {"content_type": "text", "parts": ["This should NOT be extracted."]}
                    }
                },
                "key_A": {
                    "parent": None,
                    "message": {
                        "author": {"role": "user"},
                        "content": {"content_type": "text", "parts": ["Step 1: start."]}
                    }
                },
                "key_C2_active": {
                    "parent": "key_B",
                    "message": {
                        "author": {"role": "user"},
                        "content": {"content_type": "text", "parts": ["This SHOULD be extracted."]}
                    }
                },
                "key_B": {
                    "parent": "key_A",
                    "message": {
                        "author": {"role": "assistant"},
                        "content": {"content_type": "text", "parts": ["Step 2: responding."]}
                    }
                }
            }
        }
    ]
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)


def main():
    scratch_dir = Path("backend/scratch")
    scratch_dir.mkdir(parents=True, exist_ok=True)
    
    input_path = scratch_dir / "synthetic_graph_conversations.json"
    output_path = scratch_dir / "sharegpt_graph_dataset.jsonl"
    
    print("=== Creating synthetic graph conversations.json ===")
    create_synthetic_data(input_path)
    
    print(f"Input file created at: {input_path}")
    print("--- EXACT INPUT FILE CONTENT ---")
    with open(input_path, 'r', encoding='utf-8') as f:
        print(f.read())
    print("--------------------------------\n")
    
    parser = ChatGPTParser()
    print("=== Running Graph Pipeline ===")
    raw_count, clean_count, removed_log = parser.parse(input_path, output_path)
    
    print(f"\nStats:")
    print(f"Total Raw Conversations: {raw_count}")
    print(f"Final Clean Conversations: {clean_count}")
    
    if removed_log:
        print("\nRemoved Lines / Audit Log:")
        for log in removed_log:
            print(f" - {log}")
        
    print("\n--- EXACT OUTPUT FILE CONTENT (.jsonl) ---")
    with open(output_path, 'r', encoding='utf-8') as f:
        print(f.read())
    print("------------------------------------------")

if __name__ == "__main__":
    main()
