#!/usr/bin/env python3
"""
Simple test to verify the crew setup works.
"""

import sys
import os
from pathlib import Path

# Add the src directory to the path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

try:
    from gangshit.crew import Gangshit
    
    print("🔍 Testing crew initialization...")
    
    # Create the crew instance
    crew_instance = Gangshit()
    
    print("✅ Crew class instantiated successfully")
    
    # Try to get the crew
    try:
        crew = crew_instance.gangshit_crew()
        print(f"✅ Crew created with {len(crew.agents)} agents and {len(crew.tasks)} tasks")
        
        print(f"\n🤖 Agents:")
        for i, agent in enumerate(crew.agents, 1):
            print(f"   {i}. {agent.role}")
        
        print(f"\n📋 Tasks:")
        for i, task in enumerate(crew.tasks, 1):
            print(f"   {i}. {task.description[:50]}...")
            
        print(f"\n🎉 Crew setup is working correctly!")
        
    except Exception as e:
        print(f"❌ Error creating crew: {e}")
        import traceback
        traceback.print_exc()
        
except Exception as e:
    print(f"❌ Error importing crew: {e}")
    import traceback
    traceback.print_exc()
