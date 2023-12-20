// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op/col2im.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "exceptions.hpp"
#include "ngraph/op/non_zero.hpp"
#include "ngraph/type/element_type.hpp"
#include "openvino/core/partial_shape.hpp"
#include "openvino/core/shape.hpp"
OPENVINO_SUPPRESS_DEPRECATED_START

#include "default_opset.hpp"

namespace ngraph {
namespace onnx_import {
namespace op {
namespace set_1 {

OutputVector col2im(const Node& node) {
    const auto& inputs = node.get_ng_inputs();
    const auto& columns = inputs.at(0);
    const auto& image_shape = inputs.at(1);
    const auto& kernel_shape = inputs.at(2);
    const auto& dilations = node.get_attribute_value<std::vector<std::int64_t>>(
        "dilations",
        std::vector<std::int64_t>(image_shape.get_shape().size(), 1));
    const auto& pads = node.get_attribute_value<std::vector<std::int64_t>>(
        "pads",
        std::vector<std::int64_t>(2 * image_shape.get_shape().size(), 0));
    const auto& strides = node.get_attribute_value<std::vector<std::int64_t>>(
        "strides",
        std::vector<std::int64_t>(image_shape.get_shape().size(), 1));

    CHECK_VALID_NODE(node,
                     dilations.size() == image_shape.get_shape().size(),
                     "Size of \"dilations\" attribute (got ",
                     dilations.size(),
                     ") should equal to the number of image dimensions (got ",
                     image_shape.get_shape().size(),
                     ").");
    CHECK_VALID_NODE(node,
                     pads.size() == 2 * image_shape.get_shape().size(),
                     "Size of \"pads\" attribute (got ",
                     pads.size(),
                     ") should equal to twice the number of image dimensions (got ",
                     image_shape.get_shape().size(),
                     ").");
    CHECK_VALID_NODE(node,
                     strides.size() == image_shape.get_shape().size(),
                     "Size of \"strides\" attribute (got ",
                     strides.size(),
                     ") should equal to the number of image dimensions (got ",
                     image_shape.get_shape().size(),
                     ").");

    // Reshape each column to a partial image
    // I.e., reshape columns with shape (N, C*kernel_size, num_blocks) to
    // (N, C, kernel_dim_1, kernel_dim_2, ..., kernel_dim_n, num_blocks)
    auto N = default_opset::Constant::create(element::i64, Shape(), {columns.get_shape().at(0)});
    auto kernel_shape_size =
        std::make_shared<default_opset::ReduceProd>(kernel_shape,
                                                    default_opset::Constant::create(element::i64, Shape(), {0}));
    auto C = std::make_shared<default_opset::Divide>(
        default_opset::Constant::create(element::i64, Shape(), {columns.get_shape().at(1)}),
        kernel_shape_size);
    auto partial_images_shape =
        std::make_shared<default_opset::Concat>(N,
                                                C,
                                                kernel_shape,
                                                default_opset::Constant::create(element::i64, Shape(), {-1}));
    auto partial_images = std::make_shared<default_opset::Reshape>(columns, partial_images_shape);

    // Zero-pad partial images to original padded image shape, then add them together
    // to get the original padded images.
    // TODO: Consider dilations
    // auto accumulative = default_opset::Constant::create(partial_images->get_element_type(), );
    auto partial_images_loop_param_shape = partial_images->get_output_shape(0);
    // Pop out the last dimension, i.e., the block_num dimension.
    partial_images_loop_param_shape.pop_back();
    auto partial_images_loop_param =
        std::make_shared<default_opset::Parameter>(partial_images->get_element_type(), partial_images_loop_param_shape);
    // The output's shape should be (N, C, dim_1, dim_2, ...)
    // N coule be obtained from input `columns`'s shape,
    // But C, it has to be calculated from input `block_shape` and `columns`
    // Therefore it's non-deterministic at translation time, we have to use a partial shape
    auto accumulative_input_shape = partial_images->get_output_partial_shape(0);
    accumulative_input_shape.resize(2);
    accumulative_input_shape.insert(accumulative_input_shape.end(),
                                    image_shape.get_shape().begin(),
                                    image_shape.get_shape().end());
    auto accumulative_loop_input =
        std::make_shared<default_opset::Parameter>(partial_images->get_element_type(), accumulative_input_shape);
    // auto partial_images_batch = default_opset::Gather

    const auto loop =
        std::make_shared<default_opset::Loop>(default_opset::Constant::create(element::i64, Shape(), {-1}),
                                              default_opset::Constant::create(element::boolean, Shape(), {true}));
}

}  // namespace set_1
}  // namespace op
}  // namespace onnx_import
}  // namespace ngraph
OPENVINO_SUPPRESS_DEPRECATED_END
