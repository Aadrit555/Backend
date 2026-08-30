import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

class ChatGPTParser:
    """
    Parses a standard ChatGPT conversations.json export into a ShareGPT-formatted
    jsonl dataset suitable for LLaMA-Factory and Unsloth.
    """
    
    def __init__(self):
        self.role_map = {
            "user": "human",
            "assistant": "gpt",
            "system": "system",
            "tool": "tool"
        }

    def parse(self, input_path: Path, output_path: Path) -> Tuple[int, int, List[str]]:
        """
        Parses conversations.json, cleans data, and writes to ShareGPT jsonl.
        Returns: (raw_count, clean_count, removed_reasons_list)
        """
        with open(input_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
            
        raw_count = len(raw_data)
        
        parsed_conversations = []
        removed_log = []
        seen_hashes = set()
        
        for conv in raw_data:
            mapping = conv.get("mapping", {})
            current_node_id = conv.get("current_node")
            
            # If no current_node is explicitly defined, find one that has no children (leaf node)
            # Typically ChatGPT exports include current_node, but as a fallback:
            if not current_node_id:
                # Find all nodes that are parents
                parent_ids = {node.get("parent") for node in mapping.values() if node.get("parent")}
                # Find leaves (nodes that are not parents)
                leaves = [nid for nid in mapping.keys() if nid not in parent_ids]
                if leaves:
                    current_node_id = leaves[0]
            
            if not current_node_id or current_node_id not in mapping:
                removed_log.append(f"Dropped entire conversation '{conv.get('title', 'Untitled')}' because no current_node was found.")
                continue

            # Traverse backwards from current_node to the root
            path_ids = []
            curr = current_node_id
            while curr in mapping:
                path_ids.append(curr)
                parent = mapping[curr].get("parent")
                if parent == curr:  # Cycle protection
                    break
                curr = parent
                
            # Reverse to get chronological order from root to leaf
            path_ids.reverse()
            
            msg_list = []
            for nid in path_ids:
                node = mapping.get(nid, {})
                message = node.get("message")
                if not message:
                    continue
                    
                author = message.get("author", {})
                role = author.get("role")
                if not role or role not in self.role_map:
                    continue
                    
                content = message.get("content", {})
                content_type = content.get("content_type")
                if content_type != "text":
                    continue
                    
                parts = content.get("parts", [])
                text = "".join(parts).strip()
                
                msg_list.append({
                    "role": self.role_map[role],
                    "text": text
                })
                
            # Clean: filter empty messages
            cleaned_msg_list = []
            for m in msg_list:
                if not m["text"]:
                    removed_log.append(f"Removed empty message from conversation '{conv.get('title', 'Untitled')}'")
                    continue
                
                cleaned_msg_list.append({
                    "from": m["role"],
                    "value": m["text"]
                })
                
            if not cleaned_msg_list:
                removed_log.append(f"Dropped entire conversation '{conv.get('title', 'Untitled')}' because it had no valid messages.")
                continue
                
            # Clean: Deduplication
            # Convert to a stable string to hash
            conv_str = json.dumps(cleaned_msg_list, sort_keys=True)
            import hashlib
            conv_hash = hashlib.md5(conv_str.encode()).hexdigest()
            
            if conv_hash in seen_hashes:
                removed_log.append(f"Removed duplicated conversation '{conv.get('title', 'Untitled')}' (Hash: {conv_hash})")
                continue
                
            seen_hashes.add(conv_hash)
            
            parsed_conversations.append({
                "conversations": cleaned_msg_list
            })
            
        # Write to JSONL
        with open(output_path, 'w', encoding='utf-8') as f:
            for c in parsed_conversations:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
                
        return raw_count, len(parsed_conversations), removed_log
