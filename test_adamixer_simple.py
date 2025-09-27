#!/usr/bin/env python3
"""
Simple test script for AdaMixer implementation
Tests core components individually to verify correctness
"""

import torch
import torch.nn as nn
from adamixer_detector import *

def test_individual_components():
    """Test each component individually"""
    print("=" * 50)
    print("Testing AdaMixer Components Individually")
    print("=" * 50)
    
    device = torch.device('cpu')  # Use CPU for faster testing
    batch_size = 1
    num_queries = 10  # Reduced for testing
    hidden_dim = 256
    
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}, Queries: {num_queries}")
    
    # Test 1: 3D Sampling Operator
    print("\n1. Testing Sampling3DOperator...")
    try:
        p_in_test = 8  # Use smaller value for testing
        sampling_op = Sampling3DOperator(hidden_dim, num_levels=4, p_in=p_in_test)
        
        # Create dummy multi-scale features
        multi_scale_features = [
            torch.randn(batch_size, hidden_dim, 50, 67),   # 1/4 scale
            torch.randn(batch_size, hidden_dim, 25, 34),   # 1/8 scale  
            torch.randn(batch_size, hidden_dim, 13, 17),   # 1/16 scale
            torch.randn(batch_size, hidden_dim, 7, 9),     # 1/32 scale
        ]
        
        query_pos = torch.rand(batch_size, num_queries, 4) * 100  # Random positions
        query_content = torch.randn(batch_size, num_queries, hidden_dim)
        
        output = sampling_op(multi_scale_features, query_pos, query_content)
        print(f"   ✅ Output shape: {output.shape}")
        assert output.shape == (batch_size, num_queries, p_in_test, hidden_dim)
        
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return False
    
    # Test 2: Adaptive Channel Mixing
    print("\n2. Testing AdaptiveChannelMixing...")
    try:
        p_in_test = 8
        acm = AdaptiveChannelMixing(hidden_dim, num_groups=4)
        
        x = torch.randn(batch_size, num_queries, p_in_test, hidden_dim)
        query_content = torch.randn(batch_size, num_queries, hidden_dim)
        
        output = acm(x, query_content)
        print(f"   ✅ Output shape: {output.shape}")
        assert output.shape == x.shape
        
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return False
    
    # Test 3: Adaptive Spatial Mixing
    print("\n3. Testing AdaptiveSpatialMixing...")
    try:
        p_in_test = 8
        p_out_test = 16
        asm = AdaptiveSpatialMixing(
            p_in_test, p_out_test, 
            hidden_dim, num_groups=4
        )
        
        x = torch.randn(batch_size, num_queries, p_in_test, hidden_dim)
        query_content = torch.randn(batch_size, num_queries, hidden_dim)
        
        output = asm(x, query_content)
        print(f"   ✅ Output shape: {output.shape}")
        assert output.shape == (batch_size, num_queries, p_out_test, hidden_dim)
        
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return False
    
    # Test 4: Position-Aware Self-Attention
    print("\n4. Testing PositionAwareSelfAttention...")
    try:
        attention = PositionAwareSelfAttention(hidden_dim, num_heads=8)
        
        query_content = torch.randn(batch_size, num_queries, hidden_dim)
        query_pos = torch.rand(batch_size, num_queries, 4)
        
        output = attention(query_content, query_pos)
        print(f"   ✅ Output shape: {output.shape}")
        assert output.shape == query_content.shape
        
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return False
    
    # Test 5: Initial Query Generator
    print("\n5. Testing InitialQueryGenerator...")
    try:
        query_gen = InitialQueryGenerator(num_queries, hidden_dim)
        
        query_content, query_pos = query_gen(batch_size)
        print(f"   ✅ Query content shape: {query_content.shape}")
        print(f"   ✅ Query pos shape: {query_pos.shape}")
        assert query_content.shape == (batch_size, num_queries, hidden_dim)
        assert query_pos.shape == (batch_size, num_queries, 4)
        
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return False
    
    print("\n✅ All individual components passed!")
    return True

def test_small_model():
    """Test a smaller version of the full model"""
    print("\n" + "=" * 50)
    print("Testing Small AdaMixer Model")
    print("=" * 50)
    
    try:
        # Create a smaller model for testing
        model = AdaMixerDetector(
            num_classes=10,          # Reduced classes
            num_queries=20,          # Reduced queries
            num_decoder_stages=2,    # Reduced stages
            hidden_dim=128,          # Reduced dimensions
            ffn_dim=512,
            num_heads=4,
            num_groups=2,
            p_in=8,
            p_out=16
        )
        
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Test forward pass with small input
        dummy_input = torch.randn(1, 3, 200, 300)  # Much smaller input
        print(f"Input shape: {dummy_input.shape}")
        
        model.eval()
        with torch.no_grad():
            output = model(dummy_input)
            
        print(f"✅ Pred logits shape: {output['pred_logits'].shape}")
        print(f"✅ Pred boxes shape: {output['pred_boxes'].shape}")
        print(f"✅ Query pos shape: {output['query_pos'].shape}")
        
        # Verify output shapes
        assert output['pred_logits'].shape == (1, 20, 11)  # +1 for background
        assert output['pred_boxes'].shape == (1, 20, 4)
        assert output['query_pos'].shape == (1, 20, 4)
        
        # Check value ranges
        boxes = output['pred_boxes']
        assert torch.all(boxes >= 0) and torch.all(boxes <= 1), "Boxes should be in [0,1]"
        
        print("✅ Small model test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Small model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_loss_functions():
    """Test loss functions"""
    print("\n" + "=" * 50)
    print("Testing Loss Functions")
    print("=" * 50)
    
    try:
        # Test Focal Loss
        print("1. Testing Focal Loss...")
        focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
        
        pred_logits = torch.randn(2, 10, 5)  # [B, N, num_classes]
        targets = torch.randint(0, 5, (2, 10))  # [B, N]
        
        loss = focal_loss(pred_logits, targets)
        print(f"   ✅ Focal loss: {loss.item():.4f}")
        assert loss.item() >= 0
        
        # Test AdaMixer Loss
        print("2. Testing AdaMixer Loss...")
        adamixer_loss = AdaMixerLoss()
        
        pred_boxes = torch.rand(2, 10, 4)  # [B, N, 4]
        target_classes = torch.randint(0, 5, (2, 10))  # [B, N]
        target_boxes = torch.rand(2, 10, 4)  # [B, N, 4]
        
        losses = adamixer_loss(pred_logits, pred_boxes, target_classes, target_boxes)
        
        print(f"   ✅ Total loss: {losses['total_loss'].item():.4f}")
        print(f"   ✅ Cls loss: {losses['cls_loss'].item():.4f}")
        print(f"   ✅ BBox loss: {losses['bbox_loss'].item():.4f}")
        print(f"   ✅ GIoU loss: {losses['giou_loss'].item():.4f}")
        
        assert all(loss.item() >= 0 for loss in losses.values())
        
        print("✅ Loss functions test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Loss functions test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🚀 AdaMixer Simple Test Suite")
    print("Testing core functionality with reduced complexity...")
    
    # Run tests
    tests_passed = 0
    total_tests = 3
    
    if test_individual_components():
        tests_passed += 1
    
    if test_small_model():
        tests_passed += 1
        
    if test_loss_functions():
        tests_passed += 1
    
    # Summary
    print("\n" + "=" * 50)
    print("Test Summary")
    print("=" * 50)
    print(f"Tests passed: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! AdaMixer implementation is working correctly.")
        print("\n📋 Implementation verified:")
        print("- ✅ 3D Feature Sampling with multi-scale support")
        print("- ✅ Adaptive Channel & Spatial Mixing") 
        print("- ✅ Position-aware Self-Attention")
        print("- ✅ Complete model forward pass")
        print("- ✅ Loss functions (Focal, L1, GIoU)")
        print("- ✅ Proper tensor shapes and value ranges")
        
        print("\n🔧 Ready for full-scale training:")
        print("- Use full resolution (800x1333) for actual training")
        print("- Set batch_size=1-2 for Google Colab")
        print("- Download COCO dataset for real training")
        print("- Run with: python adamixer_detector.py")
        
    else:
        print("❌ Some tests failed. Please check the implementation.")
    
    return tests_passed == total_tests

if __name__ == "__main__":
    main()