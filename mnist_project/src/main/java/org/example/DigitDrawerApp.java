package org.example;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;
import javafx.application.Application;
import javafx.embed.swing.SwingFXUtils;
import javafx.geometry.Insets;
import javafx.scene.Scene;
import javafx.scene.SnapshotParameters;
import javafx.scene.canvas.Canvas;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.chart.BarChart;
import javafx.scene.chart.CategoryAxis;
import javafx.scene.chart.NumberAxis;
import javafx.scene.chart.XYChart;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.scene.control.Spinner;
import javafx.scene.control.SpinnerValueFactory;
import javafx.scene.image.WritableImage;
import javafx.scene.layout.BorderPane;
import javafx.scene.layout.VBox;
import javafx.scene.paint.Color;
import javafx.stage.Stage;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.nio.FloatBuffer;
import java.nio.file.Paths;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.Collections;

/**
 * 升级版 JavaFX 画板：
 * 包含质心对齐预处理、推理耗时统计、概率分布可视化功能。
 */
public class DigitDrawerApp extends Application {

    private static final int CANVAS_SIZE = 280; // 显示用较大画布
    private static final int MNIST_SIZE = 28;   // 模型输入 28x28
    private static final double DEFAULT_PEN_WIDTH = 40; // 默认笔画加粗，提高识别率

    private Spinner<Integer> penWidthSpinner;
    private Canvas canvas;
    private GraphicsContext gc;

    // UI 控件
    private Label resultLabel;
    private Label timeLabel; // 显示推理耗时
    private BarChart<String, Number> probabilityChart; // 显示概率分布
    private XYChart.Series<String, Number> probSeries;

    private OrtEnvironment env;
    private OrtSession session;

    @Override
    public void start(Stage primaryStage) throws Exception {
        initOnnxSession();

        // 1. 初始化画布
        canvas = new Canvas(CANVAS_SIZE, CANVAS_SIZE);
        gc = canvas.getGraphicsContext2D();

        // 初始化 Spinner (放在 clearCanvas 之前，因为 clearCanvas 会用到它)
        penWidthSpinner = new Spinner<>();
        SpinnerValueFactory<Integer> valueFactory =
                new SpinnerValueFactory.IntegerSpinnerValueFactory(10, 60, (int) DEFAULT_PEN_WIDTH);
        penWidthSpinner.setValueFactory(valueFactory);
        penWidthSpinner.setEditable(true);
        penWidthSpinner.setMaxWidth(80);
        penWidthSpinner.valueProperty().addListener((obs, oldValue, newValue) -> {
            if (newValue != null) gc.setLineWidth(newValue.doubleValue());
        });

        clearCanvas(); // 设置白底黑笔

        // 鼠标事件
        canvas.setOnMousePressed(e -> {
            gc.beginPath();
            gc.moveTo(e.getX(), e.getY());
            gc.stroke();
        });
        canvas.setOnMouseDragged(e -> {
            gc.lineTo(e.getX(), e.getY());
            gc.stroke();
        });

        // 2. 右侧控制区
        Button clearBtn = new Button("清空画布");
        clearBtn.setOnAction(e -> {
            clearCanvas();
            resetChart(); // 清空图表
            resultLabel.setText("请在画板上写一个数字...");
            timeLabel.setText("推理耗时: - ms");
        });
        clearBtn.setMaxWidth(Double.MAX_VALUE);

        Button predictBtn = new Button("识别");
        predictBtn.setOnAction(e -> performPrediction());
        predictBtn.setMaxWidth(Double.MAX_VALUE);
        predictBtn.setStyle("-fx-background-color: #4CAF50; -fx-text-fill: white; -fx-font-weight: bold;");

        Button saveBtn = new Button("保存样本");
        saveBtn.setOnAction(e -> saveCanvasAsImage());
        saveBtn.setMaxWidth(Double.MAX_VALUE);

        // 结果展示标签
        resultLabel = new Label("请在画板上写一个数字，然后点击“识别”");
        resultLabel.setWrapText(true);
        resultLabel.setStyle("-fx-font-size: 14px; -fx-font-weight: bold; -fx-padding: 5; -fx-border-color: lightgray;");
        resultLabel.setMinHeight(40);
        resultLabel.setMaxWidth(Double.MAX_VALUE);

        timeLabel = new Label("推理耗时: - ms");
        timeLabel.setStyle("-fx-text-fill: #2196F3;");

        // 3. 初始化柱状图 (BarChart)
        CategoryAxis xAxis = new CategoryAxis();
        NumberAxis yAxis = new NumberAxis();
        yAxis.setLabel("置信度");
        yAxis.setUpperBound(1.0);

        probabilityChart = new BarChart<>(xAxis, yAxis);
        probabilityChart.setTitle("概率分布");
        probabilityChart.setAnimated(false); // 关闭动画以获得即时响应
        probabilityChart.setLegendVisible(false);
        probabilityChart.setPrefHeight(200);

        probSeries = new XYChart.Series<>();
        // 初始化 0-9 的空数据
        for (int i = 0; i < 10; i++) {
            probSeries.getData().add(new XYChart.Data<>(String.valueOf(i), 0));
        }
        probabilityChart.getData().add(probSeries);

        // 4. 布局
        VBox rightControls = new VBox(10);
        rightControls.setPadding(new Insets(15));
        rightControls.setMinWidth(300); //稍微加宽以容纳图表
        rightControls.getChildren().addAll(
                new Label("操作区:"),
                penWidthSpinner,
                clearBtn,
                saveBtn,
                predictBtn,
                new Label("识别结果:"),
                resultLabel,
                timeLabel,
                probabilityChart
        );

        BorderPane root = new BorderPane();
        root.setCenter(canvas);
        root.setRight(rightControls);
        BorderPane.setMargin(canvas, new Insets(10));
        // 给画布加个边框，看起来更清楚
        canvas.setStyle("-fx-effect: dropshadow(three-pass-box, rgba(0,0,0,0.3), 10, 0, 0, 0);");

        Scene scene = new Scene(root, CANVAS_SIZE + 340+350, CANVAS_SIZE + 150);
        primaryStage.setTitle("JavaFX 手写数字识别 (升级版)");
        primaryStage.setScene(scene);
        primaryStage.show();
    }

    @Override
    public void stop() throws Exception {
        super.stop();
        if (session != null) session.close();
        if (env != null) env.close();
    }

    private void initOnnxSession() throws Exception {
        // 请确保 model 文件夹在项目根目录下
        String modelPath = Paths.get("model", "mnist_model_1.onnx").toAbsolutePath().toString();
        env = OrtEnvironment.getEnvironment();
        session = env.createSession(modelPath, new OrtSession.SessionOptions());
    }

    private void clearCanvas() {
        gc.setFill(Color.WHITE);
        gc.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
        gc.setStroke(Color.BLACK);
        if (penWidthSpinner != null && penWidthSpinner.getValue() != null) {
            gc.setLineWidth(penWidthSpinner.getValue());
        } else {
            gc.setLineWidth(DEFAULT_PEN_WIDTH);
        }
        gc.setLineCap(javafx.scene.shape.StrokeLineCap.ROUND);
        gc.setLineJoin(javafx.scene.shape.StrokeLineJoin.ROUND);
    }

    private void resetChart() {
        for (XYChart.Data<String, Number> data : probSeries.getData()) {
            data.setYValue(0);
        }
    }

    /**
     * 执行预测并更新 UI
     */
    private void performPrediction() {
        try {
            PredictionResult result = predictDigitFromCanvas();
            if (result == null) {
                resultLabel.setText("画布为空，请先写字！");
                return;
            }

            // 1. 更新文本结果
            resultLabel.setText("预测结果： " + result.predictedDigit);

            // 2. 更新耗时
            double timeMs = result.inferenceTimeNano / 1_000_000.0;
            timeLabel.setText(String.format("推理耗时: %.2f ms", timeMs));

            // 3. 更新图表
            for (int i = 0; i < 10; i++) {
                probSeries.getData().get(i).setYValue(result.probabilities[i]);
            }

        } catch (Exception ex) {
            ex.printStackTrace();
            resultLabel.setText("错误：" + ex.getMessage());
        }
    }

    /**
     * 核心预测逻辑
     */
    private PredictionResult predictDigitFromCanvas() throws Exception {
        BufferedImage smallImg = processCanvasToMNISTImage();
        if (smallImg == null) return null;

        // 转换为 Tensor (归一化到 0-1)
        float[] data = new float[MNIST_SIZE * MNIST_SIZE];
        int idx = 0;
        for (int y = 0; y < MNIST_SIZE; y++) {
            for (int x = 0; x < MNIST_SIZE; x++) {
                int rgb = smallImg.getRGB(x, y);
                int gray = (rgb & 0xFF);
                float normalized = (255.0f - gray) / 255.0f;
                // 如果你的模型训练时用了 (x-0.1307)/0.3081，请在这里解开注释:
                // normalized = (normalized - 0.1307f) / 0.3081f;
                data[idx++] = normalized;
            }
        }

        long[] shape = new long[]{1, 1, MNIST_SIZE, MNIST_SIZE};
        FloatBuffer fb = FloatBuffer.wrap(data);
        OnnxTensor inputTensor = OnnxTensor.createTensor(env, fb, shape);

        // 计时开始
        long startTime = System.nanoTime();

        String inputName = session.getInputNames().iterator().next();
        OrtSession.Result outputs = session.run(Collections.singletonMap(inputName, inputTensor));

        // 计时结束
        long endTime = System.nanoTime();

        OnnxTensor outputTensor = (OnnxTensor) outputs.get(0);
        float[][] logits = (float[][]) outputTensor.getValue();

        // 计算概率和结果
        float[] probs = softmax(logits[0]);
        int predicted = argMax(probs);

        outputs.close();
        inputTensor.close();

        return new PredictionResult(predicted, smallImg, probs, endTime - startTime);
    }

    /**
     * 【核心改进】图像预处理：裁剪 + 缩放(20x20) + 质心对齐
     */
    private BufferedImage processCanvasToMNISTImage() {
        WritableImage writableImage = new WritableImage(CANVAS_SIZE, CANVAS_SIZE);
        canvas.snapshot(new SnapshotParameters(), writableImage);
        BufferedImage bufferedImage = SwingFXUtils.fromFXImage(writableImage, null);

        // 1. 寻找包围盒
        int minX = CANVAS_SIZE, minY = CANVAS_SIZE, maxX = -1, maxY = -1;
        int validPixelCount = 0; // 新增：有效像素计数器
        // 为了避开边缘伪影，我们不从 0 开始，而是忽略最边缘的 2 个像素
        int margin = 2;

        for (int y = margin; y < CANVAS_SIZE - margin; y++) {
            for (int x = margin; x < CANVAS_SIZE - margin; x++) {
                int rgb = bufferedImage.getRGB(x, y);
                int gray = (rgb >> 16) & 0xFF;

                // 降低阈值：只有深色笔迹(小于200)才算，避免浅灰色噪声
                if (gray < 200) {
                    if (x < minX) minX = x;
                    if (x > maxX) maxX = x;
                    if (y < minY) minY = y;
                    if (y > maxY) maxY = y;
                    validPixelCount++; // 统计像素点
                }
            }
        }

// 新增判断：只有当有效像素超过 10 个时，才认为不是噪点
        if (validPixelCount < 10) return null;

        // 2. 裁剪
        int w = maxX - minX + 1;
        int h = maxY - minY + 1;
        BufferedImage cropped = bufferedImage.getSubimage(minX, minY, w, h);

        // 3. 缩放到 20x20 (保留纵横比)
        int targetSize = 20;
        double scale = (double) targetSize / Math.max(w, h);
        int newW = (int) (w * scale);
        int newH = (int) (h * scale);

        java.awt.Image tmp = cropped.getScaledInstance(newW, newH, java.awt.Image.SCALE_SMOOTH);
        BufferedImage resized = new BufferedImage(newW, newH, BufferedImage.TYPE_BYTE_GRAY);
        Graphics2D g2 = resized.createGraphics();
        g2.drawImage(tmp, 0, 0, null);
        g2.dispose();

        // 4. 创建 28x28 画布
        BufferedImage finalImg = new BufferedImage(MNIST_SIZE, MNIST_SIZE, BufferedImage.TYPE_BYTE_GRAY);
        Graphics2D gFinal = finalImg.createGraphics();
        gFinal.setColor(java.awt.Color.WHITE);
        gFinal.fillRect(0, 0, MNIST_SIZE, MNIST_SIZE);

        // 5. 质心对齐计算
        // 先把 resize 后的图临时画在中间
        int centerX = (MNIST_SIZE - newW) / 2;
        int centerY = (MNIST_SIZE - newH) / 2;
        gFinal.drawImage(resized, centerX, centerY, null);

        // 计算质心
        double[] com = getCenterOfMass(finalImg);
        double comX = com[0];
        double comY = com[1];

        // 如果计算出质心有效，则进行平移修正
        if (!Double.isNaN(comX) && !Double.isNaN(comY)) {
            // 清空重画
            gFinal.setColor(java.awt.Color.WHITE);
            gFinal.fillRect(0, 0, MNIST_SIZE, MNIST_SIZE);

            // 计算需要的位移量：目标中心(14,14) - 当前质心
            int shiftX = (int) Math.round(MNIST_SIZE / 2.0 - comX);
            int shiftY = (int) Math.round(MNIST_SIZE / 2.0 - comY);

            // 应用位移重新绘制
            gFinal.drawImage(resized, centerX + shiftX, centerY + shiftY, null);
        }
        gFinal.dispose();

        return finalImg;
    }

    private double[] getCenterOfMass(BufferedImage img) {
        double sumX = 0, sumY = 0, sumMass = 0;
        for (int y = 0; y < img.getHeight(); y++) {
            for (int x = 0; x < img.getWidth(); x++) {
                int gray = img.getRGB(x, y) & 0xFF;
                double mass = 255.0 - gray; // 黑色是质量
                if (mass > 0) {
                    sumX += x * mass;
                    sumY += y * mass;
                    sumMass += mass;
                }
            }
        }
        if (sumMass == 0) return new double[]{Double.NaN, Double.NaN};
        return new double[]{sumX / sumMass, sumY / sumMass};
    }

    private void saveCanvasAsImage() {
        try {
            PredictionResult result = predictDigitFromCanvas(); // 重新预测一次以获取最新状态
            if (result == null) {
                resultLabel.setText("保存失败：画布空白");
                return;
            }
            String timestamp = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss"));
            String fileName = String.format("src/main/image/manual_%s_pred_%d.png", timestamp, result.predictedDigit);
            java.io.File file = new java.io.File(fileName);
            file.getParentFile().mkdirs();
            ImageIO.write(result.smallImage, "png", file);
            resultLabel.setText("样本已保存: " + file.getName());
        } catch (Exception ex) {
            ex.printStackTrace();
            resultLabel.setText("保存失败: " + ex.getMessage());
        }
    }

    private float[] softmax(float[] logits) {
        float max = Float.NEGATIVE_INFINITY;
        for (float v : logits) max = Math.max(max, v);
        float sum = 0;
        float[] probs = new float[logits.length];
        for (int i = 0; i < logits.length; i++) {
            probs[i] = (float) Math.exp(logits[i] - max);
            sum += probs[i];
        }
        for (int i = 0; i < probs.length; i++) probs[i] /= sum;
        return probs;
    }

    private int argMax(float[] arr) {
        int idx = 0;
        float max = arr[0];
        for (int i = 1; i < arr.length; i++) {
            if (arr[i] > max) { max = arr[i]; idx = i; }
        }
        return idx;
    }

    // 内部类：预测结果封装
    private static class PredictionResult {
        final int predictedDigit;
        final BufferedImage smallImage;
        final float[] probabilities;
        final long inferenceTimeNano;

        PredictionResult(int predictedDigit, BufferedImage smallImage, float[] probabilities, long inferenceTimeNano) {
            this.predictedDigit = predictedDigit;
            this.smallImage = smallImage;
            this.probabilities = probabilities;
            this.inferenceTimeNano = inferenceTimeNano;
        }
    }
}