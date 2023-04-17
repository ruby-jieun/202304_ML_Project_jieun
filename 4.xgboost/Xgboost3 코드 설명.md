---
typora-root-url: ..\0.Confusion_matrix
---

# Xgboost3 코드 설명



* `Xgboost2`에서 변경 한 부분만 설명하였습니다.





학습된 결과를 `Confusion Matrix plot`으로 만들고 .png 파일로 저장합니다.

처음엔 `plot_confusion_matrix` 함수를 사용했지만, 오류가 반복적으로 발생하여. 직접 `Confusion Matrix`를 그리는 방법으로 대체했습니다.



**변경 후 코드**

```python
import seaborn as sns

def save_confusion_matrix_plot(model, X_test, y_test, class_names, file_name):
    plt.figure(figsize=(10, 10))
    cm = confusion_matrix(y_test, model.predict(X_test))
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    ax = sns.heatmap(cm_normalized, annot=True, cmap=plt.cm.Blues, xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(rotation='vertical')
    plt.yticks(rotation='horizontal')
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(file_name)
    plt.show()

# 학습된 모델로부터 Confusion Matrix plot 저장
save_confusion_matrix_plot(xgb_model, X_test, y_test, genre_classes, 'confusion_matrix.png')
```





**데이터 셋을 test_dataset.csv로 설정한 코드 실행 결과**

```
최적의 하이퍼파라미터: {'learning_rate': 0.11712624707999042, 'max_depth': 18, 'min_child_weight': 1, 'n_estimators': 299} 'n_estimators': 299}
예측한 음악 장르: classical
정확도: 94.67%
교차 검증된 정확도: 91.17%
```

![](/confusion_matrix_test_dataset.png)





**데이터 셋을 train_dataset.csv로 설정한 코드 실행 결과**

```

```

