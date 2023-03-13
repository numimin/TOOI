Image {
	Stacks {
		Make Montage //show all stacks in one picture
	}

	Hyperstacks {}

	Color {
		Split Channels
		Merge Channels
	}
	Lookup Tables {
		...
	}
	Type {
		...
	}

	Stacks {
		Z Project
	}
}

Plugins {
	3D Viewer
}

Process {
	Filters {
		...
		Gaussian Blur
		Convolve
	}

	Morphology {
		Gray Morphology
	}
}

Склеить изображение (5 х 5)
Убрать градиенты по краям
Убрать линии склейки
Выделить контуры животных
Найти контур одного и того же животного на всех изображениях
Отследить изменение площади
Можно показать, какие части контурв растягиваются / сжимаются
Отследить перемещение
