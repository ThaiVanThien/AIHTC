# Danh sách chức năng không cần thiết có thể xóa

Sau khi phân tích cấu trúc và chức năng của dự án, dưới đây là danh sách các chức năng có thể xem xét loại bỏ để tối ưu hóa ứng dụng:

## 1. Chức năng quản trị dư thừa

- **free_port**: Chức năng giải phóng cổng có thể không cần thiết trong môi trường sản xuất và có thể gây ra vấn đề bảo mật
- **Các endpoint quản trị thừa**: Có thể rút gọn `/system/info` và `/health` thành một endpoint duy nhất

## 2. API dư thừa

- **/nlp/set-service**: Có thể loại bỏ vì việc thay đổi dịch vụ mặc định đã được xử lý qua giao diện chat
- **/nlp/config**: Chức năng hiển thị cấu hình có thể không cần thiết hoặc nên giới hạn quyền truy cập

## 3. Chức năng quản lý cache

- **/nlp/clear-cache**: Chức năng này có thể không cần thiết với người dùng cuối, nên được chuyển thành công cụ dòng lệnh cho quản trị viên

## 4. Tinh giản middleware

- **PerformanceMiddleware**: Có thể kết hợp với các middleware khác thay vì tạo một middleware riêng
- **Xử lý exception toàn cục**: Rút gọn để hạn chế hiển thị thông tin nhạy cảm

## 5. Tối ưu hóa thư viện

- **Loại bỏ thư viện không cần thiết**:
  - pandas (nếu không xử lý dữ liệu lớn)
  - openpyxl, xlrd (nếu không làm việc với file Excel)
  - pydantic-extra-types (sử dụng các loại dữ liệu cơ bản từ Pydantic)

## 6. Cải thiện quản lý mô hình

- **Tải mô hình theo yêu cầu**: Thay vì tải toàn bộ mô hình ngay khi khởi động

## 7. Mô-đun Chat

- **Endpoint /chat/documents**: Có thể loại bỏ nếu không sử dụng chức năng quản lý tài liệu
- **Chức năng so sánh từ nhiều mô hình**: Có thể không cần thiết cho người dùng cuối

## Lưu ý:
Việc loại bỏ các chức năng này nên được xem xét cẩn thận dựa trên yêu cầu cụ thể của dự án và người dùng. Một số chức năng có thể hữu ích trong giai đoạn phát triển hoặc gỡ lỗi nhưng không cần thiết trong môi trường sản xuất. 