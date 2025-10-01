import streamlit as st
import pandas as pd
import os
import io
import matplotlib.pyplot as plt


# 分类任务函数
def classification_task(data, target_variable, train_size):
    from pycaret.classification import setup, compare_models, save_model, pull, plot_model, predict_model
    setup(data=data, target=target_variable, session_id=123, normalize=True, train_size=train_size)
    best_model = compare_models()
    st.write("最佳模型：", best_model)
    save_model(best_model, 'best_classification_model')
    # 获取模型对比结果
    model_comparison = pull()
    return best_model, model_comparison


# 回归任务函数
def regression_task(data, target_variable, train_size):
    from pycaret.regression import setup, compare_models, save_model, pull, predict_model
    setup(data=data, target=target_variable, train_size=train_size)
    best_model = compare_models()
    st.write("最佳模型：", best_model)
    save_model(best_model, 'best_regression_model')
    # 获取模型对比结果
    model_comparison = pull()
    return best_model, model_comparison


# 预测函数
def prediction(model_path, prediction_file):
    if os.path.exists(f'{model_path}.pkl'):
        if 'classification' in model_path:
            from pycaret.classification import load_model, predict_model
        else:
            from pycaret.regression import load_model, predict_model
        loaded_model = load_model(model_path)
        st.write("模型已成功载入。")
        # 读取待预测数据
        if prediction_file.name.endswith('.csv'):
            prediction_data = pd.read_csv(prediction_file, encoding='utf-8-sig')
        elif prediction_file.name.endswith('.xlsx'):
            prediction_data = pd.read_excel(prediction_file, engine='openpyxl')
        predictions = predict_model(loaded_model, data=prediction_data)
        st.write("预测结果：")
        st.write(predictions)
    else:
        st.write("未找到相应的模型文件，请先训练模型。")


# 定义主函数
def main():
    st.title("机器学习算法模型平台")
    # 上传数据
    uploaded_file = st.file_uploader("上传数据集 (CSV 或 Excel格式)", type=["csv", "xlsx"])
    if uploaded_file is not None:
        # 判断文件类型并读取数据
        if uploaded_file.name.endswith('.csv'):
            data = pd.read_csv(uploaded_file, encoding='utf-8-sig')
        elif uploaded_file.name.endswith('.xlsx'):
            data = pd.read_excel(uploaded_file, engine='openpyxl')
        data = pd.DataFrame(data)
        st.markdown("数据基本内容：")
        st.write(data.head(10))
        # 选择任务类型
        task_type = st.selectbox("选择任务类型", ["分类", "回归"])
        # 选择目标变量
        target_variable = st.selectbox("选择目标变量", data.columns)
        # 输入训练集比例
        train_size = st.number_input("输入训练集比例（0 - 1之间）", min_value=0.0, max_value=1.0, value=0.7, step=0.01)
        # 初始化会话状态
        if 'best_model' not in st.session_state:
            st.session_state.best_model = None
        if'model_comparison' not in st.session_state:
            st.session_state.model_comparison = None
        # 训练模型
        if st.button("训练模型"):
            if task_type == "分类":
                st.session_state.best_model, st.session_state.model_comparison = classification_task(data, target_variable,
                                                                                                   train_size)
            else:
                st.session_state.best_model, st.session_state.model_comparison = regression_task(data, target_variable,
                                                                                                   train_size)
        # 显示模型对比数据
        if st.session_state.model_comparison is not None:
            st.write("多模型对比结果：")
            st.dataframe(st.session_state.model_comparison)
        # 载入已有模型进行预测
        if st.checkbox("载入最佳模型进行预测"):
            if task_type == "分类":
                model_path = 'best_classification_model'
            else:
                model_path = 'best_regression_model'
            # 上传待预测数据
            prediction_file = st.file_uploader("上传待预测数据", type=["csv", "xlsx"])
            if prediction_file is not None:
                prediction(model_path, prediction_file)


if __name__ == "__main__":
    main()
