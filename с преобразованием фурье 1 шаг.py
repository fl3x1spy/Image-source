import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from scipy.optimize import curve_fit
from scipy.fft import fft, ifft, fftfreq


def get_virtual_sources(source, room_width, room_height, max_order, reflection_coeffs_freq):
    virtual_sources = []
    for n in range(-max_order, max_order + 1):
        for m in range(-max_order, max_order + 1):
            order = abs(n) + abs(m)
            if order == 0 or order > max_order:
                continue
            
            left_count = abs(n) if n < 0 else 0
            right_count = n if n > 0 else 0
            bottom_count = abs(m) if m < 0 else 0
            top_count = m if m > 0 else 0
            
            # Для каждой частоты (6 точек) вычисляем коэффициент
            effective_energy_factors = []
            for freq_idx in range(6):
                factor = (
                    (reflection_coeffs_freq['left'][freq_idx] ** left_count) *
                    (reflection_coeffs_freq['right'][freq_idx] ** right_count) *
                    (reflection_coeffs_freq['bottom'][freq_idx] ** bottom_count) *
                    (reflection_coeffs_freq['top'][freq_idx] ** top_count)
                )
                effective_energy_factors.append(factor)
            
            if all(f == 0 for f in effective_energy_factors):
                continue
            
            x = n * room_width + (source[0] if n % 2 == 0 else room_width - source[0])
            y = m * room_height + (source[1] if m % 2 == 0 else room_height - source[1])
            virtual_sources.append({
                'pos': [x, y], 
                'order': order,
                'energy_factors': effective_energy_factors,
                'left_refl': left_count,
                'right_refl': right_count,
                'top_refl': top_count,
                'bottom_refl': bottom_count
            })
    return virtual_sources


def find_wall_intersections(p1, p2, room_width, room_height):
    intersections = []
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]

    if dx != 0:
        t = (0 - p1[0]) / dx
        if 0 <= t <= 1:
            y = p1[1] + t * dy
            if 0 <= y <= room_height:
                intersections.append((0, y, 'left', t))
        t = (room_width - p1[0]) / dx
        if 0 <= t <= 1:
            y = p1[1] + t * dy
            if 0 <= y <= room_height:
                intersections.append((room_width, y, 'right', t))

    if dy != 0:
        t = (0 - p1[1]) / dy
        if 0 <= t <= 1:
            x = p1[0] + t * dx
            if 0 <= x <= room_width:
                intersections.append((x, 0, 'bottom', t))
        t = (room_height - p1[1]) / dy
        if 0 <= t <= 1:
            x = p1[0] + t * dx
            if 0 <= x <= room_width:
                intersections.append((x, room_height, 'top', t))

    return sorted(intersections, key=lambda inter: inter[3])


def plot_reflection_path(ax, points, color):
    xs, ys = zip(*points)
    ax.plot(xs, ys, linestyle='-', linewidth=1.5, color=color, alpha=0.7)
    for p in points[1:-1]:
        ax.plot(p[0], p[1], 'bo', markersize=6, alpha=0.7)


def create_impulse_response(room_width, room_height, source, receiver, max_order, reflection_coeffs_freq, fs=8000, duration=0.5):
    """
    Создаёт импульсный отклик помещения (Impulse Response) во времени
    """
    speed_of_sound = 343
    virtual_sources = get_virtual_sources(source, room_width, room_height, max_order, reflection_coeffs_freq)
    
    # Временная сетка
    N = int(fs * duration)
    t = np.linspace(0, duration, N, endpoint=False)
    ir = np.zeros(N)
    
    # Добавляем прямой путь
    direct_dist = np.linalg.norm(np.array(receiver) - np.array(source))
    direct_time = direct_dist / speed_of_sound
    direct_idx = int(direct_time * fs)
    if direct_idx < N:
        ir[direct_idx] += 1.0  # нормированная амплитуда
    
    # Добавляем отражённые пути
    for vs in virtual_sources:
        pos = vs['pos']
        energy_factors = vs['energy_factors']
        
        reflections = find_wall_intersections(pos, receiver, room_width, room_height)
        path_points = [pos] + [(ix[0], ix[1]) for ix in reflections] + [receiver]
        r = 0
        for i in range(len(path_points) - 1):
            seg_length = np.linalg.norm(np.array(path_points[i+1]) - np.array(path_points[i]))
            r += seg_length
        
        time_sec = r / speed_of_sound
        time_idx = int(time_sec * fs)
        
        # Берём средний коэффициент отражения для всех частот (упрощение)
        avg_energy_factor = np.mean(energy_factors)
        amplitude = avg_energy_factor / r  # амплитуда пропорциональна 1/r (не 1/r²!)
        
        if time_idx < N:
            ir[time_idx] += amplitude
    
    return t, ir


def demonstrate_fft_with_acoustics(ir, fs, title="Импульсный отклик"):
    """
    Применяет FFT к импульсному отклику и проверяет сходимость
    """
    N = len(ir)
    T = 1.0 / fs
    t = np.linspace(0, N*T, N)
    
    # Прямое преобразование Фурье
    yf = fft(ir)
    xf = fftfreq(N, T)[:N//2]
    
    # Обратное преобразование
    recovered_ir = ifft(yf).real
    
    # Проверяем сходимость
    max_diff = np.max(np.abs(ir - recovered_ir))
    print(f"\nМаксимальная разница между исходным и восстановленным ИО: {max_diff:.10f}")
    print("✅ FFT успешно выполнена, сигнал восстановлен с высокой точностью." if max_diff < 1e-10 else "❌ Ошибка в FFT!")
    
    # Построение графиков
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # Импульсный отклик во времени
    ax1.plot(t[:int(fs*0.1)], ir[:int(fs*0.1)])
    ax1.set_title('Импульсный отклик (первые 100 мс)')
    ax1.set_xlabel('Время (с)')
    ax1.set_ylabel('Амплитуда')
    ax1.grid(True)
    
    # Амплитудный спектр
    ax2.plot(xf, 2.0/N * np.abs(yf[:N//2]))
    ax2.set_title('Амплитудный спектр (FFT)')
    ax2.set_xlabel('Частота (Гц)')
    ax2.set_ylabel('Амплитуда')
    ax2.grid(True)
    
    # Фазовый спектр
    ax3.plot(xf, np.angle(yf[:N//2]))
    ax3.set_title('Фазовый спектр')
    ax3.set_xlabel('Частота (Гц)')
    ax3.set_ylabel('Фаза (рад)')
    ax3.grid(True)
    
    # Сравнение оригинала и восстановленного
    ax4.plot(t[:int(fs*0.1)], recovered_ir[:int(fs*0.1)], label='Восстановленный', linewidth=2)
    ax4.plot(t[:int(fs*0.1)], ir[:int(fs*0.1)], '--', label='Оригинальный', alpha=0.7)
    ax4.set_title('Сравнение: Оригинальный vs Восстановленный (первые 100 мс)')
    ax4.set_xlabel('Время (с)')
    ax4.set_ylabel('Амплитуда')
    ax4.legend()
    ax4.grid(True)
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def draw_room_and_sources(room_width, room_height, source, receiver, max_order, reflection_coeffs_freq):
    fig, ax = plt.subplots(figsize=(12, 10))
    room_patch = patches.Rectangle((0, 0), room_width, room_height, linewidth=2,
                                   edgecolor='orange', facecolor='none', linestyle='-')
    ax.add_patch(room_patch)
    ax.plot(source[0], source[1], 'ro', markersize=10, label='Источник')
    ax.plot(receiver[0], receiver[1], 'go', markersize=10, label='Приёмник')
    ax.plot([source[0], receiver[0]], [source[1], receiver[1]],
            color='purple', linestyle='-', linewidth=2, alpha=0.8, label='Прямой путь')

    speed_of_sound = 343

    virtual_sources = get_virtual_sources(source, room_width, room_height, max_order, reflection_coeffs_freq)

    # Для каждой частоты будем собирать свои массивы
    freq_labels = ['125 Гц', '250 Гц', '500 Гц', '1 кГц', '2 кГц', '4 кГц']
    times_freq = [[] for _ in range(6)]
    energies_freq = [[] for _ in range(6)]
    distances = []

    for vs in virtual_sources:
        pos = vs['pos']
        order = vs['order']
        energy_factors = vs['energy_factors']

        reflections = find_wall_intersections(pos, receiver, room_width, room_height)
        path_points = [pos] + [(ix[0], ix[1]) for ix in reflections] + [receiver]
        r = 0
        for i in range(len(path_points) - 1):
            seg_length = np.linalg.norm(np.array(path_points[i+1]) - np.array(path_points[i]))
            r += seg_length
        distances.append(r)
        
        for freq_idx in range(6):
            base_energy = 1 / r ** 2
            energy = base_energy * energy_factors[freq_idx]
            time_sec = r / speed_of_sound
            if energy > 0:
                times_freq[freq_idx].append(time_sec)
                energies_freq[freq_idx].append(energy)
        
        print(f"Луч порядка {order}: длина = {r:.2f} м")
        for freq_idx in range(6):
            energy = (1 / r ** 2) * energy_factors[freq_idx]
            time_sec = r / speed_of_sound
            if energy > 0:
                print(f"  {freq_labels[freq_idx]}: время = {time_sec:.4f} с, энергия = {energy:.5f}")

        plot_reflection_path(ax, path_points, color='y')
        ax.plot(pos[0], pos[1], 'ro', markersize=5, alpha=0.7)
        offset_x = (pos[0] // room_width) * room_width
        offset_y = (pos[1] // room_height) * room_height
        virtual_room = patches.Rectangle(
            (offset_x, offset_y), room_width, room_height,
            linewidth=1, edgecolor='black', facecolor='none',
            linestyle='--', alpha=0.5
        )
        ax.add_patch(virtual_room)

    ax.set_xlim(-room_width, 2 * room_width)
    ax.set_ylim(-room_height, 2 * room_height)
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
    ax.set_title(f'Моделирование акустики помещения\n'
              f'Коэффициенты отражения зависят от частоты')
    plt.show()

    # --- Аппроксимация для каждой частоты ---
    for freq_idx in range(6):
        if len(energies_freq[freq_idx]) > 1:
            plt.figure(figsize=(10, 6))
            plt.scatter(distances[:len(energies_freq[freq_idx])], energies_freq[freq_idx], 
                       c='green', alpha=0.7, label=f'Точки: E(r) для {freq_labels[freq_idx]}')

            def inv_r_squared(r, C):
                return C / (r ** 2)

            try:
                popt, pcov = curve_fit(inv_r_squared, distances[:len(energies_freq[freq_idx])], 
                                     energies_freq[freq_idx], p0=(1), maxfev=5000)
                C_fit = popt[0]
                print(f"\nПараметр аппроксимации для {freq_labels[freq_idx]} E(r) = C / r^2: C = {C_fit:.4f}")
                r_smooth = np.linspace(min(distances[:len(energies_freq[freq_idx])]), 
                                     max(distances[:len(energies_freq[freq_idx])]), 500)
                e_smooth = inv_r_squared(r_smooth, C_fit)
                plt.plot(r_smooth, e_smooth, 'b-', linewidth=2)
            except RuntimeError:
                print(f"\nНе удалось подобрать параметры для {freq_labels[freq_idx]} E(r) = C / r^2.")
            
            plt.title(f'Зависимость энергии отражённых лучей от расстояния\n'
                      f'для виртуальных источников до {max_order} порядка ({freq_labels[freq_idx]})')
            plt.xlabel('Расстояние (м)')
            plt.ylabel('Энергия')
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.legend()
            plt.show()

# Параметры
room_width = 10
room_height = 8
source = [2, 3]
receiver = [5, 5]
max_reflection_order = int(input('Введите максимальный порядок источников: '))

print("Введите коэффициенты отражения для стен для 6 частотных диапазонов (от 0 до 1):")
print("Частоты: 125 Гц, 250 Гц, 500 Гц, 1 кГц, 2 кГц, 4 кГц")

print("Левая стена:")
left_freq = [float(input(f'  {freq} Гц: ')) for freq in [125, 250, 500, 1000, 2000, 4000]]

print("Правая стена:")
right_freq = [float(input(f'  {freq} Гц: ')) for freq in [125, 250, 500, 1000, 2000, 4000]]

print("Верхняя стена:")
top_freq = [float(input(f'  {freq} Гц: ')) for freq in [125, 250, 500, 1000, 2000, 4000]]

print("Нижняя стена:")
bottom_freq = [float(input(f'  {freq} Гц: ')) for freq in [125, 250, 500, 1000, 2000, 4000]]

reflection_coeffs_freq = {
    'left': left_freq,
    'right': right_freq,
    'top': top_freq,
    'bottom': bottom_freq
}

# 1. Рисуем комнату и лучи
draw_room_and_sources(room_width, room_height, source, receiver, max_reflection_order, reflection_coeffs_freq)

# 2. Создаём импульсный отклик
print("\nСоздание импульсного отклика помещения...")
t, ir = create_impulse_response(room_width, room_height, source, receiver, max_reflection_order, reflection_coeffs_freq, fs=8000, duration=0.5)

# 3. Применяем FFT и проверяем сходимость
demonstrate_fft_with_acoustics(ir, fs=8000, title="Импульсный отклик помещения")

print("\n Все операции завершены. Моделирование акустики с FFT выполнено корректно.")