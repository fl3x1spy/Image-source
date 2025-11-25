import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from scipy.optimize import curve_fit


def get_virtual_sources(source, room_width, room_height, max_order, reflection_coeffs):
    virtual_sources = []
    for n in range(-max_order, max_order + 1):
        for m in range(-max_order, max_order + 1):
            order = abs(n) + abs(m)
            if order == 0 or order > max_order:
                continue
            
            #Определяем эффективный коэффициент для данного виртуального источника
            #Считаем количество отражений от каждой стены
            #n - отражения по X, m - отражения по Y
            #Для n: чётные отражения от правой, нечётные от левой (в зависимости от знака)
            #Для m: чётные от нижней, нечётные от верхней (в зависимости от знака)
            
            left_count = abs(n) if n < 0 else 0
            right_count = n if n > 0 else 0
            bottom_count = abs(m) if m < 0 else 0
            top_count = m if m > 0 else 0
            
            # Учитываем коэффициенты отражения
            effective_energy_factor = (
                (reflection_coeffs['left'] ** left_count) *
                (reflection_coeffs['right'] ** right_count) *
                (reflection_coeffs['bottom'] ** bottom_count) *
                (reflection_coeffs['top'] ** top_count)
            )
            
            if effective_energy_factor == 0:
                continue  # Если энергия нулевая, пропускаем
            
            # Вычисляем координаты виртуального источника
            x = n * room_width + (source[0] if n % 2 == 0 else room_width - source[0])
            y = m * room_height + (source[1] if m % 2 == 0 else room_height - source[1])
            virtual_sources.append({
                'pos': [x, y], 
                'order': order,
                'energy_factor': effective_energy_factor,
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
    # Не отмечаем начальную и конечную точки
    for p in points[1:-1]:
        ax.plot(p[0], p[1], 'bo', markersize=6, alpha=0.7)


def draw_room_and_sources(room_width, room_height, source, receiver, max_order, reflection_coeffs):
    fig, ax = plt.subplots(figsize=(12, 10))
    room_patch = patches.Rectangle((0, 0), room_width, room_height, linewidth=2,
                                   edgecolor='orange', facecolor='none', linestyle='-')
    ax.add_patch(room_patch)
    ax.plot(source[0], source[1], 'ro', markersize=10, label='Источник')
    ax.plot(receiver[0], receiver[1], 'go', markersize=10, label='Приёмник')
    ax.plot([source[0], receiver[0]], [source[1], receiver[1]],
            color='purple', linestyle='-', linewidth=2, alpha=0.8, label='Прямой путь')

    speed_of_sound = 343

    virtual_sources = get_virtual_sources(source, room_width, room_height, max_order, reflection_coeffs)

    times = []
    energies = []
    distances = []

    for vs in virtual_sources:
        pos = vs['pos']
        order = vs['order']
        energy_factor = vs['energy_factor']

        reflections = find_wall_intersections(pos, receiver, room_width, room_height)
        path_points = [pos] + [(ix[0], ix[1]) for ix in reflections] + [receiver]
        r = 0
        for i in range(len(path_points) - 1):
            seg_length = np.linalg.norm(np.array(path_points[i+1]) - np.array(path_points[i]))
            r += seg_length
        base_energy = 1 / r ** 2
        energy = base_energy * energy_factor  # Учитываем коэффициенты отражения
        time_sec = r / speed_of_sound
        print(f"Луч порядка {order}: длина = {r:.2f} м, время = {time_sec:.4f} с, его энергия = {energy:.5f}")
        # Фильтр: сохраняем только если энергия > 0
        if energy > 0:
            times.append(time_sec)
            energies.append(energy)
            distances.append(r)

        color = np.random.rand(3,)
        plot_reflection_path(ax, path_points, color=color)
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
              f'Коэф. отражения левой стены: {reflection_coeffs["left"]}, правой: {reflection_coeffs["right"]}, '
              f'верхней: {reflection_coeffs["top"]}, нижней: {reflection_coeffs["bottom"]}')
    plt.show()

    #Аппроксимация: E(r) = C / r^2
    plt.figure(figsize=(10, 6))
    plt.scatter(distances, energies, c='green', alpha=0.7, label='Точки: E(r)')

    def inv_r_squared(r, C):
        return C / (r ** 2)

    if len(distances) > 1:
        try:
            popt, pcov = curve_fit(inv_r_squared, distances, energies, p0=(1), maxfev=5000)
            C_fit = popt[0]
            print(f"\nПараметр аппроксимации для E(r) = C / r^2: C = {C_fit:.4f}")
            r_smooth = np.linspace(min(distances), max(distances), 500)
            e_smooth = inv_r_squared(r_smooth, C_fit)
            plt.plot(r_smooth, e_smooth, 'b-', linewidth=2)
        except RuntimeError:
            print("\nНе удалось подобрать параметры для E(r) = C / r^2.")
    plt.title(f'Зависимость энергии отражённых лучей от расстояния\n'
              f'для виртуальных источников до {max_order} порядка')
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
print("Введите коэффициенты отражения для стен (от 0 до 1):")
left_refl = float(input('  Левая стена: '))
right_refl = float(input('  Правая стена: '))
top_refl = float(input('  Верхняя стена: '))
bottom_refl = float(input('  Нижняя стена: '))

reflection_coeffs = {
    'left': left_refl,
    'right': right_refl,
    'top': top_refl,
    'bottom': bottom_refl
}

draw_room_and_sources(room_width, room_height, source, receiver, max_reflection_order, reflection_coeffs)